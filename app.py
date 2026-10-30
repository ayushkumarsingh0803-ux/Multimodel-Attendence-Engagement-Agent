"""
app.py — Main Flask application entry point for the
Multimodal Attendance & Engagement Agent.
"""

import os
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_from_directory,
    flash,
    redirect,
    url_for,
)
from werkzeug.utils import secure_filename

from config import Config
from src.storage.database import (
    init_db,
    db,
    Session as SessionModel,
    Attendee,
    EngagementRecord,
    TeachingSuggestion,
    save_attendance,
    save_engagement,
    save_suggestion,
    get_session,
)
from src.storage.file_manager import FileManager
from src.ai_services.gemini_service import GeminiService
from src.ai_services.tts_service import TTSService
from src.ai_services.code_generator import CodeGenerator
from src.vision.face_tracker import FaceTracker
from src.vision.engagement_analyzer import EngagementAnalyzer
from src.audio.audio_processor import AudioProcessor
from src.fusion.multimodal_fusion import MultimodalFusion
from src.utils.route_handlers import (
    validate_session_request,
    format_attendance_response,
    format_engagement_response,
    handle_file_upload,
)
from src.utils.genai_utils import build_session_context

# ── Application Factory ───────────────────────────────────────────────────────


def create_app(config: type[Config] = Config) -> Flask:
    """
    Create and configure the Flask application.

    Args:
        config: Configuration class to use.

    Returns:
        Configured Flask app instance.
    """
    app = Flask(__name__)
    app.config.from_object(config)
    app.secret_key = config.SECRET_KEY

    # Ensure required directories exist
    config.ensure_directories()

    # Validate configuration and log warnings
    warnings = config.validate()
    for warning in warnings:
        logging.warning(warning)

    # Initialise SQLAlchemy
    db.init_app(app)
    with app.app_context():
        init_db()

    # Initialise services
    gemini = GeminiService(config.GEMINI_API_KEY, config.MODEL_NAME)
    tts = TTSService(config.AUDIO_FOLDER)
    code_gen = CodeGenerator(gemini, config.CODE_FOLDER)
    file_mgr = FileManager(config.UPLOAD_FOLDER, config.AUDIO_FOLDER, config.CODE_FOLDER)
    face_tracker = FaceTracker()
    engagement_analyzer = EngagementAnalyzer()
    audio_processor = AudioProcessor()
    fusion = MultimodalFusion(config.VISUAL_WEIGHT, config.AUDIO_WEIGHT)

    # ── Page Routes ───────────────────────────────────────────────────────────

    @app.route("/")
    def dashboard():
        """Render the main dashboard page."""
        return render_template("index.html", page="dashboard")

    @app.route("/attendance")
    def attendance():
        """Render the attendance management page."""
        sessions = SessionModel.query.order_by(SessionModel.start_time.desc()).all()
        return render_template("attendance.html", page="attendance", sessions=sessions)

    @app.route("/engagement")
    def engagement():
        """Render the engagement analytics page."""
        sessions = SessionModel.query.order_by(SessionModel.start_time.desc()).all()
        return render_template("engagement.html", page="engagement", sessions=sessions)

    @app.route("/session")
    def session():
        """Render the live session control page."""
        return render_template("session.html", page="session")

    # ── API: Session Management ───────────────────────────────────────────────

    @app.route("/api/start-session", methods=["POST"])
    def api_start_session():
        """
        Start a new classroom session.

        Request JSON:
            name (str): Human-readable session name.

        Returns:
            JSON with session_id and status.
        """
        data = request.get_json(force=True, silent=True) or {}
        valid, error = validate_session_request(data)
        if not valid:
            return jsonify({"error": error}), 400

        # Check for already-active session
        active = SessionModel.query.filter_by(status="active").first()
        if active:
            return jsonify(
                {"error": "A session is already active.", "session_id": active.id}
            ), 409

        new_session = SessionModel(
            name=data.get("name", f"Session {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}"),
            start_time=datetime.now(timezone.utc),
            status="active",
        )
        db.session.add(new_session)
        db.session.commit()

        return jsonify(
            {
                "status": "started",
                "session_id": new_session.id,
                "name": new_session.name,
                "start_time": new_session.start_time.isoformat(),
            }
        ), 201

    @app.route("/api/stop-session", methods=["POST"])
    def api_stop_session():
        """
        Stop the currently active session.

        Request JSON:
            session_id (int, optional): Specific session to stop.

        Returns:
            JSON with final session summary.
        """
        data = request.get_json(force=True, silent=True) or {}
        session_id = data.get("session_id")

        if session_id:
            target = get_session(session_id)
        else:
            target = SessionModel.query.filter_by(status="active").first()

        if not target:
            return jsonify({"error": "No active session found."}), 404

        target.end_time = datetime.now(timezone.utc)
        target.status = "completed"
        db.session.commit()

        # Compute summary
        records = EngagementRecord.query.filter_by(session_id=target.id).all()
        avg_score = (
            sum(r.fused_score for r in records) / len(records) if records else 0
        )
        attendee_count = Attendee.query.filter_by(session_id=target.id).count()

        return jsonify(
            {
                "status": "stopped",
                "session_id": target.id,
                "name": target.name,
                "end_time": target.end_time.isoformat(),
                "total_attendees": attendee_count,
                "average_engagement": round(avg_score, 1),
            }
        )

    # ── API: Video Processing ─────────────────────────────────────────────────

    @app.route("/api/upload-video", methods=["POST"])
    def api_upload_video():
        """
        Upload and process a classroom video file.

        Form data:
            file: Video file (mp4, avi, mov, mkv, webm).
            session_id (int): Session to associate the video with.

        Returns:
            JSON with processing results (attendance + engagement).
        """
        filepath, error = handle_file_upload(request, config.ALLOWED_VIDEO_EXTENSIONS)
        if error:
            return jsonify({"error": error}), 400

        session_id = request.form.get("session_id", type=int)
        if not session_id:
            return jsonify({"error": "session_id is required."}), 400

        target_session = get_session(session_id)
        if not target_session:
            return jsonify({"error": f"Session {session_id} not found."}), 404

        # Save uploaded file
        filename = secure_filename(request.files.get("file").filename)
        saved_path = file_mgr.save_upload(request.files["file"], filename)

        # Process video with OpenCV
        import cv2 as cv

        cap = cv.VideoCapture(saved_path)
        frame_interval = 30  # Analyse every 30th frame (~1 fps for 30fps video)
        frame_count = 0
        attendance_map: dict[str, dict] = {}
        engagement_scores: list[float] = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            if frame_count % frame_interval != 0:
                continue

            # Vision analysis
            faces = face_tracker.detect_faces(frame)
            presence = face_tracker.track_presence(faces, session_id)
            vis_score = engagement_analyzer.analyze_frame(frame, faces)
            audio_level = 0.5  # Placeholder: real audio extracted from video separately
            fused = fusion.fuse_signals(vis_score, audio_level, datetime.now(timezone.utc))

            engagement_scores.append(fused["fused_score"])

            # Update attendance map
            for face in faces:
                fid = face["id"]
                now_ts = datetime.now(timezone.utc)
                if fid not in attendance_map:
                    attendance_map[fid] = {
                        "face_id": fid,
                        "first_seen": now_ts,
                        "last_seen": now_ts,
                        "engagement_scores": [],
                    }
                attendance_map[fid]["last_seen"] = now_ts
                attendance_map[fid]["engagement_scores"].append(vis_score)

            # Persist engagement record
            level = engagement_analyzer.classify_engagement_level(fused["fused_score"])
            save_engagement(
                session_id=session_id,
                visual_score=vis_score,
                audio_level=audio_level,
                fused_score=fused["fused_score"],
                level=level,
            )

        cap.release()

        # Persist attendance records
        for face_id, info in attendance_map.items():
            duration = (info["last_seen"] - info["first_seen"]).total_seconds()
            avg_eng = (
                sum(info["engagement_scores"]) / len(info["engagement_scores"])
                if info["engagement_scores"]
                else 0
            )
            save_attendance(
                session_id=session_id,
                face_id=face_id,
                first_seen=info["first_seen"],
                last_seen=info["last_seen"],
                total_duration=duration,
                engagement_avg=avg_eng,
            )

        overall_avg = (
            sum(engagement_scores) / len(engagement_scores) if engagement_scores else 0
        )

        return jsonify(
            {
                "status": "processed",
                "file": filename,
                "frames_analysed": frame_count // frame_interval,
                "total_attendees": len(attendance_map),
                "average_engagement": round(overall_avg, 1),
            }
        )

    # ── API: Report Generation ────────────────────────────────────────────────

    @app.route("/api/generate-report", methods=["POST"])
    def api_generate_report():
        """
        Generate an AI-powered session report using Gemini.

        Request JSON:
            session_id (int): Session to generate the report for.

        Returns:
            JSON with markdown report text.
        """
        data = request.get_json(force=True, silent=True) or {}
        session_id = data.get("session_id")
        if not session_id:
            return jsonify({"error": "session_id is required."}), 400

        target_session = get_session(session_id)
        if not target_session:
            return jsonify({"error": f"Session {session_id} not found."}), 404

        records = EngagementRecord.query.filter_by(session_id=session_id).all()
        attendees = Attendee.query.filter_by(session_id=session_id).all()

        session_data = {
            "id": target_session.id,
            "name": target_session.name,
            "start_time": target_session.start_time.isoformat() if target_session.start_time else None,
            "end_time": target_session.end_time.isoformat() if target_session.end_time else None,
            "total_attendees": len(attendees),
            "total_engagement_records": len(records),
            "average_engagement": (
                sum(r.fused_score for r in records) / len(records) if records else 0
            ),
        }

        engagement_history = [
            {"timestamp": r.timestamp.isoformat(), "score": r.fused_score, "level": r.level}
            for r in records[:50]  # limit to 50 records for prompt size
        ]

        context = build_session_context(session_data, engagement_history)
        report = gemini.generate_session_report(session_data)

        return jsonify({"status": "ok", "report": report, "context": context})

    # ── API: Data Endpoints ───────────────────────────────────────────────────

    @app.route("/api/attendance-data")
    def api_attendance_data():
        """
        Return attendance records as JSON.

        Query params:
            session_id (int, optional): Filter by session.
            limit (int, optional): Max records (default 100).
        """
        session_id = request.args.get("session_id", type=int)
        limit = request.args.get("limit", 100, type=int)

        query = Attendee.query
        if session_id:
            query = query.filter_by(session_id=session_id)

        records = query.order_by(Attendee.first_seen.desc()).limit(limit).all()
        sessions = SessionModel.query.order_by(SessionModel.start_time.desc()).all()

        return jsonify(format_attendance_response(records, sessions))

    @app.route("/api/engagement-data")
    def api_engagement_data():
        """
        Return engagement records as JSON.

        Query params:
            session_id (int, optional): Filter by session.
            limit (int, optional): Max records (default 200).
        """
        session_id = request.args.get("session_id", type=int)
        limit = request.args.get("limit", 200, type=int)

        query = EngagementRecord.query
        if session_id:
            query = query.filter_by(session_id=session_id)

        records = query.order_by(EngagementRecord.timestamp.asc()).limit(limit).all()
        return jsonify(format_engagement_response(records))

    @app.route("/api/session-status")
    def api_session_status():
        """Return status of the currently active session (if any)."""
        active = SessionModel.query.filter_by(status="active").first()
        if active:
            last_record = (
                EngagementRecord.query.filter_by(session_id=active.id)
                .order_by(EngagementRecord.timestamp.desc())
                .first()
            )
            return jsonify(
                {
                    "active": True,
                    "session_id": active.id,
                    "name": active.name,
                    "start_time": active.start_time.isoformat(),
                    "current_engagement": last_record.fused_score if last_record else 0,
                    "current_level": last_record.level if last_record else "unknown",
                }
            )
        return jsonify({"active": False})

    # ── API: AI Features ──────────────────────────────────────────────────────

    @app.route("/api/teaching-suggestions", methods=["POST"])
    def api_teaching_suggestions():
        """
        Generate AI teaching suggestions for a session.

        Request JSON:
            session_id (int): Session to analyse.

        Returns:
            JSON with list of suggestion strings.
        """
        data = request.get_json(force=True, silent=True) or {}
        session_id = data.get("session_id")
        if not session_id:
            return jsonify({"error": "session_id is required."}), 400

        records = EngagementRecord.query.filter_by(session_id=session_id).all()
        if not records:
            return jsonify({"suggestions": ["No engagement data available yet."]})

        engagement_data = {
            "session_id": session_id,
            "records": [
                {"timestamp": r.timestamp.isoformat(), "score": r.fused_score, "level": r.level}
                for r in records
            ],
            "average_score": sum(r.fused_score for r in records) / len(records),
        }

        suggestions = gemini.generate_teaching_suggestions(engagement_data)

        # Persist suggestions
        for suggestion_text in suggestions:
            save_suggestion(
                session_id=session_id,
                suggestion_text=suggestion_text,
                context="auto-generated",
            )

        return jsonify({"status": "ok", "suggestions": suggestions})

    @app.route("/api/text-to-speech", methods=["POST"])
    def api_text_to_speech():
        """
        Convert text to speech and return the audio file URL.

        Request JSON:
            text (str): Text to synthesize.
            filename (str, optional): Output filename without extension.

        Returns:
            JSON with audio_url.
        """
        data = request.get_json(force=True, silent=True) or {}
        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "text is required."}), 400

        filename = data.get("filename", f"announcement_{int(datetime.now(timezone.utc).timestamp())}")
        filepath = tts.synthesize(text, filename)
        audio_url = f"/audio/{Path(filepath).name}"

        return jsonify({"status": "ok", "audio_url": audio_url, "filepath": filepath})

    @app.route("/api/generate-code", methods=["POST"])
    def api_generate_code():
        """
        Generate an analysis script using Gemini.

        Request JSON:
            requirements (str): Description of what the script should do.
            filename (str, optional): Output filename.

        Returns:
            JSON with code and file URL.
        """
        data = request.get_json(force=True, silent=True) or {}
        requirements = data.get("requirements", "").strip()
        if not requirements:
            return jsonify({"error": "requirements is required."}), 400

        filename = data.get("filename", f"script_{int(datetime.now(timezone.utc).timestamp())}.py")
        code = code_gen.generate_analysis_script(requirements)
        filepath = code_gen.save_script(code, filename)

        return jsonify({"status": "ok", "filename": filename, "code": code, "filepath": filepath})

    # ── Static File Serving ───────────────────────────────────────────────────

    @app.route("/audio/<path:filename>")
    def serve_audio(filename: str):
        """Serve generated audio files."""
        return send_from_directory(config.AUDIO_FOLDER, filename)

    @app.route("/uploads/<path:filename>")
    def serve_upload(filename: str):
        """Serve uploaded files."""
        return send_from_directory(config.UPLOAD_FOLDER, filename)

    # ── Error Handlers ────────────────────────────────────────────────────────

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Not found."}), 404

    @app.errorhandler(413)
    def too_large(e):
        return jsonify({"error": "File too large. Maximum size is 500 MB."}), 413

    @app.errorhandler(500)
    def server_error(e):
        return jsonify({"error": "Internal server error.", "detail": str(e)}), 500

    return app


# ── Entry Point ───────────────────────────────────────────────────────────────

app = create_app(Config)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG if Config.DEBUG else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)
