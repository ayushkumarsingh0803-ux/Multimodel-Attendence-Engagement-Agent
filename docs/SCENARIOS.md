# 🎯 Operational Scenarios: Multimodal Attendance & Engagement Agent

This document explains the end-to-end technical execution flow for the three core classroom scenarios supported by the agent.

---

## 🕒 Scenario 1: Automated Attendance Tracking

### Context & Challenge
Traditional attendance takes 5–15 minutes of lecture time and is prone to proxy attendance or manual logging errors. Single-frame snapshots often suffer from false positives due to visitors or students briefly entering the classroom.

### Technical Workflow
1. **Video Ingestion**: The system receives a live video feed or uploaded recording via `/api/upload-video`.
2. **Face & Pose Detection**: OpenCV and MediaPipe detect all facial regions frame-by-frame.
3. **Temporal Sustained Tracking**:
   - Each detected individual is tracked across sequential frames using centroid tracking and feature representations.
   - Presence is registered only when sustained beyond the configurable presence threshold (e.g., minimum 3 minutes of continuous presence).
4. **Attendance Record Generation**:
   - Computes `first_seen`, `last_seen`, and `total_duration` (minutes).
   - Generates unique persistent session tokens.
   - Stores attendance records directly in SQLite via SQLAlchemy (`Attendee` model).
5. **Output**:
   - Instant live table in the web UI.
   - One-click CSV export with confidence scores.

---

## 📊 Scenario 2: Engagement and Session Dynamics Analysis

### Context & Challenge
Presence alone does not guarantee attention. Lecturers cannot easily track engagement drops or recognize when pacing causes cognitive fatigue across a large cohort.

### Technical Workflow
1. **Visual Signal Processing**:
   - **Head Pose Estimation**: Measures yaw, pitch, and roll to assess forward gaze versus distraction (e.g., looking down at phones).
   - **Body Dynamics & Inactivity**: Tracks movement vectors to flag prolonged inactivity or excessive disengagement.
   - Produces a normalized Visual Engagement Score ($V_s \in [0, 100]$).
2. **Audio Signal Processing**:
   - **Voice Activity Detection (VAD)**: Differentiates lecture speech, peer-to-peer discussion, and dead silence.
   - **Interaction Level**: Measures acoustic energy and turn-taking frequency.
   - Produces a normalized Audio Interaction Score ($A_s \in [0, 100]$).
3. **Multimodal Fusion**:
   - Fuses visual and acoustic scores:
     $$\text{Engagement Score} = (w_v \cdot V_s) + (w_a \cdot A_s)$$
     *(Default weights: $w_v = 0.60$, $w_a = 0.40$)*
4. **Dynamics Curve Visualization**:
   - Aggregates scores into 60-second moving average intervals.
   - Emits time-series data to Chart.js in the instructor's dashboard.

---

## 🤖 Scenario 3: AI-Driven Teaching Suggestions

### Context & Challenge
Instructors benefit from having an intelligent, non-intrusive co-pilot to suggest interventions when attention lags.

### Technical Workflow
1. **Context Extraction**:
   - When requested or upon detecting a statistically significant engagement dip (>25% drop over 5 minutes), the agent compiles the session state.
2. **Prompt Synthesis**:
   - `src/utils/genai_utils.py` formats the engagement timeline, peak interaction points, and classroom dynamics into a structured prompt.
3. **Gemini 2.0 Flash Reasoning**:
   - Calls Google Gemini 2.0 Flash via `src/ai_services/gemini_service.py`.
   - The model reasons over the temporal dynamics and outputs context-aware suggestions:
     - *e.g., "Pacing Alert: Attention dropped after 25 minutes of continuous lecturing. Suggest running a 2-minute peer think-pair-share activity."*
4. **Assistive Dispatch**:
   - Insights display on the instructor's private dashboard.
   - Optional text-to-speech engine (`src/ai_services/tts_service.py`) can synthesize audio announcements or break notifications.
