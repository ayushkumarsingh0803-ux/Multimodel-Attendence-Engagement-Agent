/**
 * main.js — Main JavaScript controller for Multimodal Attendance Agent
 *
 * Contains:
 *   - apiFetch()             — Centralised fetch wrapper with error handling
 *   - DashboardController   — Dashboard stats, trend chart, session table
 *   - AttendanceController  — Attendance table, CSV export, filters
 *   - EngagementController  — Engagement charts, AI suggestions, reports
 */

'use strict';

// ── API Fetch Wrapper ─────────────────────────────────────────────────────────

/**
 * Centralised fetch helper with JSON handling and error toasts.
 *
 * @param {string} url - Endpoint URL.
 * @param {RequestInit} [options] - Fetch options.
 * @returns {Promise<any>} Parsed JSON response body.
 * @throws {Error} On network error or non-2xx response.
 */
async function apiFetch(url, options = {}) {
    const defaultHeaders = { 'Content-Type': 'application/json' };
    const config = {
        ...options,
        headers: { ...defaultHeaders, ...(options.headers || {}) },
    };

    try {
        const response = await fetch(url, config);
        const data = await response.json().catch(() => ({}));

        if (!response.ok) {
            const msg = data.error || `HTTP ${response.status}`;
            showToast(msg, 'danger');
            throw new Error(msg);
        }

        return data;
    } catch (err) {
        if (!(err.message && err.message.startsWith('HTTP'))) {
            showToast(`Network error: ${err.message}`, 'danger');
        }
        throw err;
    }
}

/**
 * Show a Bootstrap-style toast notification.
 *
 * @param {string} message - Toast body text.
 * @param {'success'|'danger'|'warning'|'info'} type - Bootstrap colour variant.
 */
function showToast(message, type = 'info') {
    // Create toast container if it doesn't exist
    let container = document.getElementById('toastContainer');
    if (!container) {
        container = document.createElement('div');
        container.id = 'toastContainer';
        container.className = 'toast-container position-fixed top-0 end-0 p-3';
        container.style.zIndex = '9999';
        document.body.appendChild(container);
    }

    const id = `toast-${Date.now()}`;
    const iconMap = { success: 'check-circle', danger: 'exclamation-triangle', warning: 'exclamation-circle', info: 'info-circle' };
    const icon = iconMap[type] || 'info-circle';

    const toastEl = document.createElement('div');
    toastEl.id = id;
    toastEl.className = `toast align-items-center text-bg-${type} border-0`;
    toastEl.setAttribute('role', 'alert');
    toastEl.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">
                <i class="bi bi-${icon} me-2"></i>${message}
            </div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
        </div>`;

    container.appendChild(toastEl);
    const toast = new bootstrap.Toast(toastEl, { delay: 4000 });
    toast.show();
    toastEl.addEventListener('hidden.bs.toast', () => toastEl.remove());
}

/**
 * Format seconds into a human-readable duration string.
 *
 * @param {number} seconds - Duration in seconds.
 * @returns {string} e.g. "1h 23m" or "45m 12s"
 */
function formatDuration(seconds) {
    if (!seconds || seconds < 0) return '—';
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    if (h > 0) return `${h}h ${m}m`;
    if (m > 0) return `${m}m ${s}s`;
    return `${s}s`;
}

/**
 * Render an engagement level badge HTML string.
 *
 * @param {string} level - "high" | "medium" | "low"
 * @returns {string} HTML span
 */
function engagementBadge(level) {
    const label = (level || 'unknown').charAt(0).toUpperCase() + (level || 'unknown').slice(1);
    return `<span class="engagement-badge ${level || ''}">${label}</span>`;
}

// ── Chart Helpers ─────────────────────────────────────────────────────────────

const ChartDefaults = {
    fontFamily: "'Inter', system-ui, sans-serif",
    gridColor: 'rgba(0,0,0,0.05)',
    engagementColors: {
        high:   '#12b76a',
        medium: '#f79009',
        low:    '#f04438',
    },
};

/**
 * Build a Chart.js line chart configuration for engagement scores.
 *
 * @param {string[]} labels - X-axis labels (timestamps or indices).
 * @param {number[]} data - Y-axis engagement scores.
 * @param {string} [label] - Dataset label.
 * @returns {object} Chart.js config object.
 */
function buildEngagementLineConfig(labels, data, label = 'Engagement Score') {
    return {
        type: 'line',
        data: {
            labels,
            datasets: [{
                label,
                data,
                borderColor: '#1a6dff',
                backgroundColor: 'rgba(26,109,255,0.08)',
                fill: true,
                tension: 0.4,
                pointRadius: data.length > 50 ? 0 : 3,
                pointHoverRadius: 5,
                borderWidth: 2,
            }],
        },
        options: {
            responsive: true,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: ctx => `${ctx.dataset.label}: ${ctx.parsed.y?.toFixed(1)}`,
                    },
                },
            },
            scales: {
                x: {
                    grid: { color: ChartDefaults.gridColor },
                    ticks: { maxTicksLimit: 10, font: { size: 11 } },
                },
                y: {
                    min: 0, max: 100,
                    grid: { color: ChartDefaults.gridColor },
                    ticks: { font: { size: 11 }, callback: v => `${v}%` },
                },
            },
        },
    };
}

// ── DashboardController ───────────────────────────────────────────────────────

const DashboardController = {
    _trendChart: null,

    /**
     * Initialise the dashboard: load stats and render trend chart.
     */
    async init() {
        await this.refresh();
    },

    /**
     * Refresh all dashboard data from the API.
     */
    async refresh() {
        try {
            const [attData, engData] = await Promise.all([
                apiFetch('/api/attendance-data?limit=500'),
                apiFetch('/api/engagement-data?limit=200'),
            ]);
            this._renderStats(attData, engData);
            this._renderSessionsTable(attData.sessions || []);
            this.loadTrendChart(engData);
        } catch (e) { /* already toasted */ }
    },

    /**
     * Render the 4 stat cards.
     */
    _renderStats(attData, engData) {
        const sessions = attData.sessions || [];
        const activeSessions = sessions.filter(s => s.status === 'active');
        const totalAttendees = attData.count || 0;

        const avgEng = engData.summary?.avg_fused
            ? `${engData.summary.avg_fused.toFixed(1)}%`
            : '—';

        this._set('statTotalSessions', sessions.length);
        this._set('statTotalAttendees', totalAttendees);
        this._set('statAvgEngagement', avgEng);
        this._set('statActiveSessions', activeSessions.length);
        this._set('activeSessionName', activeSessions[0]?.name || '');
    },

    /**
     * Render the recent sessions table.
     */
    _renderSessionsTable(sessions) {
        const tbody = document.getElementById('recentSessionsTbody');
        const count = document.getElementById('recentSessionsCount');
        if (!tbody) return;

        const recent = sessions.slice(0, 10);
        count && (count.textContent = `${sessions.length} total`);

        if (!recent.length) {
            tbody.innerHTML = `<tr><td colspan="7" class="text-center py-4 text-muted">
                No sessions yet. <a href="/session">Start your first session</a>.
            </td></tr>`;
            return;
        }

        tbody.innerHTML = recent.map(s => {
            const start = s.start_time ? new Date(s.start_time).toLocaleString() : '—';
            const duration = (s.start_time && s.end_time)
                ? formatDuration((new Date(s.end_time) - new Date(s.start_time)) / 1000)
                : '—';
            const statusClass = { active: 'bg-danger', completed: 'bg-success', cancelled: 'bg-secondary' }[s.status] || 'bg-light';
            return `
            <tr>
                <td class="fw-semibold">${s.name}</td>
                <td>${start}</td>
                <td>${duration}</td>
                <td>—</td>
                <td>—</td>
                <td><span class="badge ${statusClass}">${s.status}</span></td>
                <td>
                    <a href="/attendance" class="btn btn-sm btn-outline-primary">View</a>
                </td>
            </tr>`;
        }).join('');
    },

    /**
     * Load and render the 7-day engagement trend chart.
     */
    async loadTrendChart(engData = null) {
        const canvas = document.getElementById('engagementTrendChart');
        if (!canvas) return;

        if (!engData) {
            try { engData = await apiFetch('/api/engagement-data?limit=200'); }
            catch (e) { return; }
        }

        const timeline = engData.timeline || [];
        const labels = timeline.map(p => {
            const d = new Date(p.t);
            return isNaN(d) ? p.t : d.toLocaleDateString([], { month: 'short', day: 'numeric' });
        });
        const values = timeline.map(p => p.y);

        if (this._trendChart) { this._trendChart.destroy(); }

        this._trendChart = new Chart(canvas, buildEngagementLineConfig(labels, values, 'Engagement Trend'));
    },

    _set(id, value) {
        const el = document.getElementById(id);
        if (el) el.textContent = value;
    },
};

// ── AttendanceController ──────────────────────────────────────────────────────

const AttendanceController = {
    _allRecords: [],
    _filteredRecords: [],
    _page: 1,
    _pageSize: 20,

    async init() {
        await this.loadSession('');
    },

    async loadSession(sessionId) {
        try {
            const url = sessionId
                ? `/api/attendance-data?session_id=${sessionId}&limit=500`
                : '/api/attendance-data?limit=500';
            const data = await apiFetch(url);
            this._allRecords = data.records || [];
            this._filteredRecords = [...this._allRecords];
            this._page = 1;
            this._renderSummary();
            this._renderTable();
        } catch (e) { /* toasted */ }
    },

    applyFilters() {
        const from = document.getElementById('filterDateFrom')?.value;
        const to   = document.getElementById('filterDateTo')?.value;

        this._filteredRecords = this._allRecords.filter(r => {
            if (from && r.first_seen && r.first_seen < from) return false;
            if (to   && r.first_seen && r.first_seen > to + 'T23:59:59') return false;
            return true;
        });
        this._page = 1;
        this._renderSummary();
        this._renderTable();
    },

    searchTable(query) {
        const q = query.trim().toLowerCase();
        this._filteredRecords = q
            ? this._allRecords.filter(r => r.face_id.toLowerCase().includes(q))
            : [...this._allRecords];
        this._page = 1;
        this._renderTable();
    },

    _renderSummary() {
        const recs = this._filteredRecords;
        const total = recs.length;
        const present = recs.filter(r => r.status === 'present').length;
        const avgDur = total ? (recs.reduce((a, r) => a + (r.total_duration_min || 0), 0) / total).toFixed(1) : '—';
        const avgEng = total ? (recs.reduce((a, r) => a + (r.engagement_avg || 0), 0) / total).toFixed(1) + '%' : '—';

        const s = id => document.getElementById(id);
        s('sumTotal') && (s('sumTotal').textContent = total);
        s('sumPresent') && (s('sumPresent').textContent = present);
        s('sumAvgDuration') && (s('sumAvgDuration').textContent = avgDur);
        s('sumAvgEngagement') && (s('sumAvgEngagement').textContent = avgEng);
    },

    _renderTable() {
        const tbody = document.getElementById('attendanceTbody');
        const info  = document.getElementById('tableInfo');
        if (!tbody) return;

        const start = (this._page - 1) * this._pageSize;
        const pageRecs = this._filteredRecords.slice(start, start + this._pageSize);

        if (!pageRecs.length) {
            tbody.innerHTML = `<tr><td colspan="8" class="text-center py-5 text-muted">
                No attendance records match the current filter.
            </td></tr>`;
            info && (info.textContent = 'No records.');
            return;
        }

        tbody.innerHTML = pageRecs.map((r, i) => `
        <tr>
            <td>${start + i + 1}</td>
            <td><code>${r.face_id}</code></td>
            <td>${r.session_id || '—'}</td>
            <td>${r.first_seen ? new Date(r.first_seen).toLocaleTimeString() : '—'}</td>
            <td>${r.last_seen  ? new Date(r.last_seen).toLocaleTimeString()  : '—'}</td>
            <td>${formatDuration(r.total_duration_s)}</td>
            <td>
                <div class="d-flex align-items-center gap-2">
                    <div class="progress flex-grow-1" style="height:6px;">
                        <div class="progress-bar ${r.engagement_avg >= 70 ? 'bg-success' : r.engagement_avg >= 40 ? 'bg-warning' : 'bg-danger'}"
                             style="width:${Math.min(r.engagement_avg || 0, 100)}%"></div>
                    </div>
                    <span class="small">${(r.engagement_avg || 0).toFixed(0)}%</span>
                </div>
            </td>
            <td><span class="badge ${r.status === 'present' ? 'bg-success' : 'bg-secondary'}">${r.status}</span></td>
        </tr>`).join('');

        info && (info.textContent = `Showing ${start + 1}–${Math.min(start + this._pageSize, this._filteredRecords.length)} of ${this._filteredRecords.length} records`);
    },

    exportCSV() {
        const recs = this._filteredRecords;
        if (!recs.length) { showToast('No records to export.', 'warning'); return; }

        const headers = ['Face ID', 'Session ID', 'First Seen', 'Last Seen', 'Duration (min)', 'Engagement %', 'Status'];
        const rows = recs.map(r => [
            r.face_id, r.session_id,
            r.first_seen ? new Date(r.first_seen).toLocaleString() : '',
            r.last_seen  ? new Date(r.last_seen).toLocaleString()  : '',
            (r.total_duration_min || 0).toFixed(2),
            (r.engagement_avg || 0).toFixed(1),
            r.status,
        ]);

        const csv = [headers, ...rows].map(row => row.join(',')).join('\n');
        const blob = new Blob([csv], { type: 'text/csv' });
        const url  = URL.createObjectURL(blob);
        const a    = document.createElement('a');
        a.href = url;
        a.download = `attendance_${Date.now()}.csv`;
        a.click();
        URL.revokeObjectURL(url);
        showToast('Attendance exported as CSV.', 'success');
    },
};

// ── EngagementController ──────────────────────────────────────────────────────

const EngagementController = {
    _timelineChart: null,
    _pieChart: null,
    _comparisonChart: null,
    _currentSessionId: null,
    _reportText: '',

    init() {
        // No-op on load; wait for session selection
    },

    async loadSession(sessionId) {
        this._currentSessionId = sessionId;
        if (!sessionId) return;

        try {
            const data = await apiFetch(`/api/engagement-data?session_id=${sessionId}&limit=500`);
            this._renderOverviewCards(data.summary || {});
            this._renderTimelineChart(data.timeline || []);
            this._renderPieChart(data.summary?.level_counts || {});
            this._renderComparisonChart(data.records || []);
        } catch (e) { /* toasted */ }
    },

    refresh() { this.loadSession(this._currentSessionId); },

    _renderOverviewCards(summary) {
        const set = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
        set('engAvgScore', summary.avg_fused != null ? `${summary.avg_fused.toFixed(1)}%` : '—');
        set('engPeakScore',   summary.max_fused  != null ? `${summary.max_fused.toFixed(1)}%`  : '—');
        set('engValleyScore', summary.min_fused  != null ? `${summary.min_fused.toFixed(1)}%`  : '—');
        set('engTotalRecords', document.getElementById('engTotalRecords')?.dataset?.count || '—');
    },

    _renderTimelineChart(timeline) {
        const canvas = document.getElementById('engagementTimelineChart');
        if (!canvas) return;

        const labels = timeline.map(p => {
            if (!p.t) return '';
            const d = new Date(p.t);
            return isNaN(d) ? p.t : d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        });
        const values = timeline.map(p => p.y);

        if (this._timelineChart) this._timelineChart.destroy();
        this._timelineChart = new Chart(canvas, buildEngagementLineConfig(labels, values, 'Fused Engagement'));

        document.getElementById('engTotalRecords') && (document.getElementById('engTotalRecords').textContent = timeline.length);
    },

    _renderPieChart(levelCounts) {
        const canvas = document.getElementById('levelDistributionChart');
        if (!canvas) return;

        if (this._pieChart) this._pieChart.destroy();
        this._pieChart = new Chart(canvas, {
            type: 'doughnut',
            data: {
                labels: ['High', 'Medium', 'Low'],
                datasets: [{
                    data: [levelCounts.high || 0, levelCounts.medium || 0, levelCounts.low || 0],
                    backgroundColor: ['#12b76a', '#f79009', '#f04438'],
                    borderWidth: 0,
                }],
            },
            options: {
                cutout: '65%',
                plugins: { legend: { display: false } },
            },
        });
    },

    _renderComparisonChart(records) {
        const canvas = document.getElementById('signalComparisonChart');
        if (!canvas || !records.length) return;

        const labels = records.map((_, i) => i + 1);
        const visual = records.map(r => r.visual_score);
        const audio  = records.map(r => (r.audio_level || 0) * 100);

        if (this._comparisonChart) this._comparisonChart.destroy();
        this._comparisonChart = new Chart(canvas, {
            type: 'line',
            data: {
                labels,
                datasets: [
                    {
                        label: 'Visual Score',
                        data: visual,
                        borderColor: '#1a6dff',
                        backgroundColor: 'rgba(26,109,255,0.05)',
                        fill: true, tension: 0.4, pointRadius: 0, borderWidth: 2,
                    },
                    {
                        label: 'Audio Level (%)',
                        data: audio,
                        borderColor: '#12b76a',
                        backgroundColor: 'rgba(18,183,106,0.05)',
                        fill: true, tension: 0.4, pointRadius: 0, borderWidth: 2,
                    },
                ],
            },
            options: {
                responsive: true,
                interaction: { mode: 'index', intersect: false },
                plugins: { legend: { position: 'top', labels: { boxWidth: 12, font: { size: 11 } } } },
                scales: {
                    x: { display: false },
                    y: { min: 0, max: 100, grid: { color: 'rgba(0,0,0,0.05)' }, ticks: { callback: v => `${v}%` } },
                },
            },
        });
    },

    async fetchSuggestions() {
        if (!this._currentSessionId) { showToast('Please select a session first.', 'warning'); return; }

        const loadingEl  = document.getElementById('suggestionsLoading');
        const listEl     = document.getElementById('suggestionsList');
        const btn        = document.getElementById('fetchSuggestionsBtn');

        loadingEl?.classList.remove('d-none');
        listEl && (listEl.innerHTML = '');
        btn && (btn.disabled = true);

        try {
            const data = await apiFetch('/api/teaching-suggestions', {
                method: 'POST',
                body: JSON.stringify({ session_id: parseInt(this._currentSessionId) }),
            });

            const suggestions = data.suggestions || [];
            listEl && (listEl.innerHTML = suggestions.length
                ? `<div class="list-group list-group-flush">
                    ${suggestions.map((s, i) => `
                        <div class="list-group-item border-0 d-flex gap-3 px-0">
                            <div class="flex-shrink-0">
                                <span class="badge bg-primary rounded-pill">${i + 1}</span>
                            </div>
                            <div>${s}</div>
                        </div>`).join('')}
                   </div>`
                : '<p class="text-muted">No suggestions generated.</p>');

            showToast(`${suggestions.length} suggestions generated.`, 'success');
        } catch (e) { /* toasted */ }
        finally {
            loadingEl?.classList.add('d-none');
            btn && (btn.disabled = false);
        }
    },

    async generateReport() {
        if (!this._currentSessionId) { showToast('Please select a session first.', 'warning'); return; }

        const modal = new bootstrap.Modal(document.getElementById('reportModal'));
        const body  = document.getElementById('reportModalBody');
        modal.show();

        try {
            const data = await apiFetch('/api/generate-report', {
                method: 'POST',
                body: JSON.stringify({ session_id: parseInt(this._currentSessionId) }),
            });
            this._reportText = data.report || 'No report generated.';
            // Simple markdown to HTML conversion
            body && (body.innerHTML = `<div class="report-content">${this._markdownToHtml(this._reportText)}</div>`);
        } catch (e) {
            body && (body.innerHTML = '<div class="text-danger">Failed to generate report.</div>');
        }
    },

    copyReport() {
        navigator.clipboard.writeText(this._reportText).then(() => showToast('Report copied!', 'success'));
    },

    _markdownToHtml(md) {
        return md
            .replace(/^### (.+)$/gm, '<h5>$1</h5>')
            .replace(/^## (.+)$/gm, '<h4>$1</h4>')
            .replace(/^# (.+)$/gm, '<h3>$1</h3>')
            .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.+?)\*/g, '<em>$1</em>')
            .replace(/^- (.+)$/gm, '<li>$1</li>')
            .replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>')
            .replace(/\n\n/g, '</p><p>')
            .replace(/^(.+)$/gm, (m) => m.startsWith('<') ? m : `<p>${m}</p>`);
    },
};
