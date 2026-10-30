/**
 * session.js — Live session page controllers
 *
 * Contains:
 *   - SessionController     — Start/stop sessions, poll status
 *   - VideoUploader         — Drag-drop upload, progress, processing
 *   - LiveEngagementGauge   — Chart.js doughnut gauge
 *   - SessionLog            — Auto-scrolling session event log
 *   - TTSController         — Text-to-speech announcements
 */

'use strict';

// ── SessionController ─────────────────────────────────────────────────────────

const SessionController = {
    _sessionId: null,
    _startTimestamp: null,
    _pollInterval: null,
    _durationInterval: null,

    /**
     * Initialise the session controller by checking for an active session.
     */
    async init() {
        try {
            const status = await apiFetch('/api/session-status');
            if (status.active) {
                this._sessionId = status.session_id;
                this._startTimestamp = new Date(status.start_time);
                this._showActivePanel(status.name);
                this._startPolling();
                SessionLog.append(`Resumed active session: "${status.name}"`, 'info');
            }
        } catch (e) { /* ignore */ }
    },

    /**
     * Start a new classroom session.
     */
    async start() {
        const nameInput = document.getElementById('sessionNameInput');
        const name = nameInput?.value.trim() || 'Unnamed Session';

        const btn = document.getElementById('startSessionBtn');
        if (btn) { btn.disabled = true; btn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Starting...'; }

        try {
            const data = await apiFetch('/api/start-session', {
                method: 'POST',
                body: JSON.stringify({ name }),
            });

            this._sessionId = data.session_id;
            this._startTimestamp = new Date(data.start_time);
            this._showActivePanel(data.name);
            this._startPolling();

            // Enable process button if file is selected
            const processBtn = document.getElementById('processVideoBtn');
            if (processBtn && VideoUploader._selectedFile) processBtn.disabled = false;

            SessionLog.append(`Session started: "${data.name}" (ID: ${data.session_id})`, 'success');
            showToast(`Session "${data.name}" started.`, 'success');
        } catch (e) {
            if (btn) { btn.disabled = false; btn.innerHTML = '<i class="bi bi-play-fill me-2"></i>Start Session'; }
        }
    },

    /**
     * Stop the current active session.
     */
    async stop() {
        if (!this._sessionId) { showToast('No active session.', 'warning'); return; }

        const confirmed = confirm('Are you sure you want to stop the current session?');
        if (!confirmed) return;

        try {
            const data = await apiFetch('/api/stop-session', {
                method: 'POST',
                body: JSON.stringify({ session_id: this._sessionId }),
            });

            this._stopPolling();
            this._showIdlePanel();

            SessionLog.append(`Session stopped. Attendees: ${data.total_attendees}, Avg Engagement: ${data.average_engagement}%`, 'warning');
            showToast(`Session "${data.name}" completed.`, 'success');

            this._sessionId = null;
            this._startTimestamp = null;
        } catch (e) { /* toasted */ }
    },

    _showActivePanel(name) {
        document.getElementById('noSessionPanel')?.classList.add('d-none');
        document.getElementById('activeSessionPanel')?.classList.remove('d-none');
        const nameDisplay = document.getElementById('activeSessionNameDisplay');
        if (nameDisplay) nameDisplay.textContent = name;
        this._startDurationCounter();
    },

    _showIdlePanel() {
        document.getElementById('activeSessionPanel')?.classList.add('d-none');
        document.getElementById('noSessionPanel')?.classList.remove('d-none');
        clearInterval(this._durationInterval);
        const durEl = document.getElementById('rtDuration');
        if (durEl) durEl.textContent = '—';
    },

    _startDurationCounter() {
        clearInterval(this._durationInterval);
        this._durationInterval = setInterval(() => {
            if (!this._startTimestamp) return;
            const elapsed = (Date.now() - this._startTimestamp.getTime()) / 1000;
            const el = document.getElementById('rtDuration');
            if (el) el.textContent = formatDuration(elapsed);
        }, 1000);
    },

    _startPolling() {
        clearInterval(this._pollInterval);
        this._pollInterval = setInterval(async () => {
            try {
                const status = await apiFetch('/api/session-status');
                if (status.active) {
                    this._updateRealTimeStats(status);
                }
            } catch (e) { /* ignore polling errors */ }
        }, 5000);
    },

    _stopPolling() { clearInterval(this._pollInterval); },

    _updateRealTimeStats(status) {
        const score = status.current_engagement || 0;
        const level = status.current_level || 'unknown';

        const scoreEl = document.getElementById('rtEngagementScore');
        const levelEl = document.getElementById('rtEngagementLevel');

        if (scoreEl) scoreEl.textContent = `${score.toFixed(1)}%`;
        if (levelEl) levelEl.innerHTML = engagementBadge(level);

        // Update gauge
        LiveEngagementGauge.update(score, level);
    },
};

// ── VideoUploader ─────────────────────────────────────────────────────────────

const VideoUploader = {
    _selectedFile: null,

    /**
     * Initialise drag-and-drop event listeners.
     */
    init() {
        const dropzone = document.getElementById('uploadDropzone');
        const fileInput = document.getElementById('videoFileInput');

        if (!dropzone) return;

        dropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropzone.classList.add('drag-over');
        });

        dropzone.addEventListener('dragleave', () => {
            dropzone.classList.remove('drag-over');
        });

        dropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropzone.classList.remove('drag-over');
            const file = e.dataTransfer?.files[0];
            if (file) this._handleFile(file);
        });

        dropzone.addEventListener('click', (e) => {
            if (!e.target.closest('button')) {
                fileInput?.click();
            }
        });

        fileInput?.addEventListener('change', () => {
            const file = fileInput.files?.[0];
            if (file) this._handleFile(file);
        });
    },

    _handleFile(file) {
        const allowedTypes = ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-matroska', 'video/webm', 'video/x-msvideo'];
        const maxSize = 500 * 1024 * 1024;

        if (!allowedTypes.includes(file.type) && !file.name.match(/\.(mp4|avi|mov|mkv|webm)$/i)) {
            showToast('Invalid file type. Please upload a video file.', 'danger');
            return;
        }

        if (file.size > maxSize) {
            showToast('File is too large. Maximum size is 500 MB.', 'danger');
            return;
        }

        this._selectedFile = file;

        // Show file info
        document.getElementById('fileSelectedInfo')?.classList.remove('d-none');
        const nameEl = document.getElementById('selectedFileName');
        if (nameEl) nameEl.textContent = `${file.name} (${(file.size / 1024 / 1024).toFixed(1)} MB)`;

        // Enable process button if session is active
        const processBtn = document.getElementById('processVideoBtn');
        if (processBtn) processBtn.disabled = !SessionController._sessionId;

        SessionLog.append(`File selected: ${file.name}`, 'info');
    },

    clearFile() {
        this._selectedFile = null;
        document.getElementById('fileSelectedInfo')?.classList.add('d-none');
        document.getElementById('processVideoBtn') && (document.getElementById('processVideoBtn').disabled = true);
        const fileInput = document.getElementById('videoFileInput');
        if (fileInput) fileInput.value = '';
    },

    /**
     * Upload and process the selected video file.
     */
    async upload() {
        if (!this._selectedFile) { showToast('No file selected.', 'warning'); return; }
        if (!SessionController._sessionId) { showToast('Start a session first.', 'warning'); return; }

        const progressContainer = document.getElementById('uploadProgressContainer');
        const progressBar       = document.getElementById('uploadProgressBar');
        const progressLabel     = document.getElementById('uploadProgressLabel');
        const progressPct       = document.getElementById('uploadProgressPct');
        const processBtn        = document.getElementById('processVideoBtn');
        const resultsEl         = document.getElementById('processingResults');

        progressContainer?.classList.remove('d-none');
        resultsEl?.classList.add('d-none');
        if (processBtn) { processBtn.disabled = true; processBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Processing...'; }

        SessionLog.append(`Uploading: ${this._selectedFile.name}...`, 'info');

        const formData = new FormData();
        formData.append('file', this._selectedFile);
        formData.append('session_id', SessionController._sessionId);

        try {
            // Simulate progress (real XHR progress is complex with fetch)
            let pct = 0;
            const fakeProgress = setInterval(() => {
                pct = Math.min(pct + 5, 85);
                if (progressBar) progressBar.style.width = `${pct}%`;
                if (progressPct) progressPct.textContent = `${pct}%`;
            }, 300);

            const response = await fetch('/api/upload-video', {
                method: 'POST',
                body: formData,
            });
            clearInterval(fakeProgress);

            const data = await response.json();

            if (progressBar) progressBar.style.width = '100%';
            if (progressPct) progressPct.textContent = '100%';
            if (progressLabel) progressLabel.textContent = 'Complete!';

            if (!response.ok) throw new Error(data.error || 'Upload failed');

            // Show results
            resultsEl?.classList.remove('d-none');
            const s = id => document.getElementById(id);
            s('resFrames')     && (s('resFrames').textContent     = data.frames_analysed || '—');
            s('resAttendees')  && (s('resAttendees').textContent  = data.total_attendees || '—');
            s('resEngagement') && (s('resEngagement').textContent = data.average_engagement != null ? `${data.average_engagement}%` : '—');

            SessionLog.append(`Processing complete. Attendees: ${data.total_attendees}, Avg Engagement: ${data.average_engagement}%`, 'success');
            showToast('Video processed successfully!', 'success');
        } catch (e) {
            SessionLog.append(`Upload failed: ${e.message}`, 'error');
        } finally {
            if (processBtn) {
                processBtn.disabled = false;
                processBtn.innerHTML = '<i class="bi bi-cpu me-2"></i>Process Video';
            }
            setTimeout(() => progressContainer?.classList.add('d-none'), 2000);
        }
    },
};

// ── LiveEngagementGauge ───────────────────────────────────────────────────────

const LiveEngagementGauge = {
    _chart: null,

    /**
     * Initialise the doughnut gauge chart.
     *
     * @param {string} canvasId - ID of the <canvas> element.
     */
    init(canvasId) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return;

        this._chart = new Chart(canvas, {
            type: 'doughnut',
            data: {
                datasets: [{
                    data: [0, 100],
                    backgroundColor: ['#6b7a99', '#e8ecf2'],
                    borderWidth: 0,
                    circumference: 240,
                    rotation: -120,
                }],
            },
            options: {
                cutout: '80%',
                responsive: true,
                plugins: { legend: { display: false }, tooltip: { enabled: false } },
                animation: { duration: 600 },
            },
        });
    },

    /**
     * Update the gauge to reflect a new engagement score.
     *
     * @param {number} score - Engagement score in [0, 100].
     * @param {string} level - "high" | "medium" | "low"
     */
    update(score, level) {
        if (!this._chart) return;

        const clamped = Math.max(0, Math.min(100, score));
        const colorMap = { high: '#12b76a', medium: '#f79009', low: '#f04438' };
        const color = colorMap[level] || '#6b7a99';

        this._chart.data.datasets[0].data = [clamped, 100 - clamped];
        this._chart.data.datasets[0].backgroundColor = [color, '#e8ecf2'];
        this._chart.update();

        const displayEl = document.getElementById('gaugeScoreDisplay');
        if (displayEl) {
            displayEl.textContent = `${clamped.toFixed(1)}%`;
            displayEl.style.color = color;
        }
    },
};

// ── SessionLog ────────────────────────────────────────────────────────────────

const SessionLog = {
    _containerId: 'sessionLog',

    /**
     * Initialise the log container.
     *
     * @param {string} containerId - ID of the log container div.
     */
    init(containerId) {
        this._containerId = containerId;
    },

    /**
     * Append a new entry to the session log and auto-scroll.
     *
     * @param {string} message - Log message text.
     * @param {'info'|'success'|'warning'|'error'} type - Entry type.
     */
    append(message, type = 'info') {
        const container = document.getElementById(this._containerId);
        if (!container) return;

        const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        const entry = document.createElement('div');
        entry.className = `log-entry log-entry--${type}`;
        entry.innerHTML = `<span class="log-time">${time}</span><span class="log-msg">${this._escape(message)}</span>`;

        container.appendChild(entry);

        // Auto-scroll to bottom
        container.scrollTop = container.scrollHeight;

        // Limit to 200 entries
        const entries = container.querySelectorAll('.log-entry');
        if (entries.length > 200) entries[0].remove();
    },

    /**
     * Clear all log entries.
     */
    clear() {
        const container = document.getElementById(this._containerId);
        if (container) container.innerHTML = '';
        this.append('Log cleared.', 'info');
    },

    _escape(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    },
};

// ── TTSController ─────────────────────────────────────────────────────────────

const TTSController = {
    /**
     * Initialise TTS controller (sets up enter-key shortcut).
     */
    init() {
        const textarea = document.getElementById('ttsText');
        textarea?.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'Enter') {
                e.preventDefault();
                this.send();
            }
        });
    },

    /**
     * Send the text in the TTS textarea to the API and play the result.
     */
    async send() {
        const textarea = document.getElementById('ttsText');
        const text = textarea?.value.trim();

        if (!text) { showToast('Please enter some text to speak.', 'warning'); return; }

        SessionLog.append(`TTS: "${text.substring(0, 60)}${text.length > 60 ? '...' : ''}"`, 'info');

        try {
            const data = await apiFetch('/api/text-to-speech', {
                method: 'POST',
                body: JSON.stringify({ text }),
            });

            const playerContainer = document.getElementById('ttsAudioPlayer');
            const audio = document.getElementById('ttsAudio');

            if (playerContainer && audio) {
                playerContainer.classList.remove('d-none');
                audio.src = data.audio_url;
                audio.play().catch(() => {
                    showToast('Audio ready — click play to listen.', 'info');
                });
            }

            SessionLog.append(`TTS audio ready: ${data.audio_url}`, 'success');
            showToast('Announcement synthesised!', 'success');
        } catch (e) { /* toasted */ }
    },
};
