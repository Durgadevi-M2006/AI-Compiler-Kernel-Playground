/**
 * AI Compiler & Kernel Playground - Session History & Data Export Controller
 */

let currentSessionHistory = [];

document.addEventListener('DOMContentLoaded', () => {
  initHistoryControls();
});

function initHistoryControls() {
  const btnExportCsv = document.getElementById('btn-history-export-csv');
  const btnExportJson = document.getElementById('btn-history-export-json');
  const btnClear = document.getElementById('btn-history-clear');

  if (btnExportCsv) {
    btnExportCsv.addEventListener('click', () => {
      window.location.href = '/api/export/csv';
    });
  }

  if (btnExportJson) {
    btnExportJson.addEventListener('click', () => {
      window.location.href = '/api/export/json';
    });
  }

  if (btnClear) {
    btnClear.addEventListener('click', async () => {
      if (confirm('Are you sure you want to clear the current session run history?')) {
        try {
          await fetch('/api/history', { method: 'DELETE' });
          await loadSessionHistory();
        } catch (err) {
          console.error('Failed to clear history:', err);
        }
      }
    });
  }
}

/**
 * Loads session run records and renders the table
 */
async function loadSessionHistory() {
  const tbody = document.getElementById('session-history-table-body');
  if (!tbody) return;

  try {
    const res = await fetch('/api/history');
    const data = await res.json();
    currentSessionHistory = data.history || [];

    if (currentSessionHistory.length === 0) {
      tbody.innerHTML = `
        <tr>
          <td colspan="10" style="text-align: center; color: var(--text-muted); padding: 24px;">
            No session runs recorded yet. Execute code in the Playground or run a benchmark to log results!
          </td>
        </tr>
      `;
      return;
    }

    tbody.innerHTML = currentSessionHistory.map(row => {
      const paramStr = typeof row.parameters === 'object' ? JSON.stringify(row.parameters) : String(row.parameters || '{}');
      const isVerified = row.correctness === true || row.correctness === 'VERIFIED';
      const badgeClass = isVerified ? 'badge-green' : 'badge-amber';
      const statusText = isVerified ? 'VERIFIED' : 'CHECK';

      return `
        <tr>
          <td class="font-mono text-muted">#${row.id}</td>
          <td style="font-size: 11.5px; color: var(--text-muted);">${row.timestamp}</td>
          <td><strong style="color: var(--text-primary);">${row.experiment}</strong></td>
          <td><span class="badge badge-muted">${row.language}</span></td>
          <td class="font-mono" style="font-size: 11.5px; color: var(--text-secondary); max-width: 140px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="${paramStr}">${paramStr}</td>
          <td class="font-mono" style="color: var(--accent-cyan); font-weight: 600;">${row.execution_time_ms} ms</td>
          <td><span class="badge badge-green">${row.speedup || '1.0x'}</span></td>
          <td><span class="badge ${badgeClass}">${statusText}</span></td>
          <td style="font-size: 11.5px; color: var(--text-secondary); max-width: 180px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="${row.notes || ''}">${row.notes || 'Normal run'}</td>
          <td>
            <div style="display: flex; gap: 6px;">
              <button class="btn btn-sm btn-outline-blue" onclick="openHistoryItem(${row.id})" title="Restore program state into Playground">Open</button>
              <button class="btn btn-sm btn-outline" style="color: var(--accent-rose); border-color: var(--accent-rose-border); padding: 4px 8px;" onclick="deleteHistoryItem(${row.id})" title="Delete record">✕</button>
            </div>
          </td>
        </tr>
      `;
    }).join('');

  } catch (err) {
    console.error('Failed to load session history:', err);
  }
}

/**
 * Restores a recorded program into Playground and re-analyzes
 */
window.openHistoryItem = async function(id) {
  const item = currentSessionHistory.find(r => r.id === id);
  if (!item) return;

  const code = item.code || '';
  const lang = (item.language || 'python').toLowerCase();
  const textarea = document.getElementById('playground-code-textarea');
  const langSelect = document.getElementById('playground-lang-select');

  if (langSelect) {
    langSelect.value = lang.includes('mojo') ? 'mojo' : 'python';
  }

  if (textarea && code) {
    textarea.value = code;
    textarea.dispatchEvent(new Event('input'));
  }

  // Update central ActiveProgramContext
  if (window.ActiveProgramContext) {
    window.ActiveProgramContext.program_id = `history_${id}`;
    window.ActiveProgramContext.title = item.experiment || 'Restored Program';
    window.ActiveProgramContext.language = lang.includes('mojo') ? 'mojo' : 'python';
    window.ActiveProgramContext.code = code || textarea?.value || '';
    window.ActiveProgramContext.analyzed = false;
  }

  window.navigateToView('playground');

  // Trigger analysis for restored program
  if (typeof window.analyzeCurrentCode === 'function') {
    await window.analyzeCurrentCode();
  }
};

/**
 * Deletes a single history record by ID
 */
window.deleteHistoryItem = async function(id) {
  try {
    await fetch(`/api/history/${id}`, { method: 'DELETE' });
    await loadSessionHistory();
  } catch (err) {
    console.error('Failed to delete history item:', err);
  }
};

window.loadSessionHistory = loadSessionHistory;

