/**
 * AI Compiler & Kernel Playground - Master Application Controller
 * Handles ActiveProgramContext (Single Source of Truth), SPA View Routing, 
 * Sidebar Navigation, Health Checks, Modals & Keyboard Shortcuts
 */

// Global Single Source of Truth for the active user program
window.ActiveProgramContext = {
  program_id: 'vec_add',
  title: 'Vector Addition',
  language: 'python',
  code: '',
  analyzed: false,
  workload: 'Vector Operations',
  detected_operations: ['Input Vector A, B', 'Scalar Loop', 'Scalar Add', 'List Append'],
  analysis: null,
  benchmark: null
};

let currentActiveView = 'dashboard';
let hostEnvironment = null;

document.addEventListener('DOMContentLoaded', async () => {
  initViewRouter();
  initSidebarControls();
  initHelpModal();
  initDashboardActions();
  await checkSystemStatus();
  window.updateActiveProgramUI();
});

/**
 * Updates all Active Program Banners and toggles Empty States across views
 */
window.updateActiveProgramUI = function() {
  const ctx = window.ActiveProgramContext;
  const isAnalyzed = !!(ctx && ctx.analyzed && ctx.analysis);
  const title = ctx.title || 'Custom Kernel';
  const lang = (ctx.language || 'python').toUpperCase();

  // 1. Update Playground header info
  const playTitle = document.getElementById('playground-active-title');
  const playLang = document.getElementById('playground-active-lang');
  const playStatus = document.getElementById('playground-active-status');
  const playOps = document.getElementById('playground-detected-ops-pill');

  if (playTitle) playTitle.textContent = title;
  if (playLang) playLang.textContent = lang;
  if (playStatus) {
    if (isAnalyzed) {
      playStatus.textContent = 'Analyzed ✓';
      playStatus.className = 'badge badge-green';
    } else {
      playStatus.textContent = 'Not Analyzed';
      playStatus.className = 'badge badge-amber';
    }
  }
  if (playOps) {
    const ops = ctx.detected_operations && ctx.detected_operations.length > 0 
      ? ctx.detected_operations.join(' ➔ ') 
      : 'Input ➔ Compute ➔ Output';
    playOps.innerHTML = `Detected: <span style="color: var(--accent-cyan); font-weight: 500;">${ops}</span>`;
  }

  // 2. Update Analyze view banner
  const anzTitle = document.getElementById('analyze-active-title');
  const anzLang = document.getElementById('analyze-active-lang');
  const anzStatus = document.getElementById('analyze-active-status');
  if (anzTitle) anzTitle.textContent = title;
  if (anzLang) anzLang.textContent = lang;
  if (anzStatus) {
    anzStatus.textContent = isAnalyzed ? 'Analyzed ✓' : 'Not Analyzed';
    anzStatus.className = isAnalyzed ? 'badge badge-green' : 'badge badge-amber';
  }

  // 3. Update Graph view banner & empty state
  const grpTitle = document.getElementById('graph-active-title');
  const grpLang = document.getElementById('graph-active-lang');
  const grpStatus = document.getElementById('graph-active-status');
  const grpEmpty = document.getElementById('graph-empty-state');
  const grpContent = document.getElementById('graph-main-content');

  if (grpTitle) grpTitle.textContent = title;
  if (grpLang) grpLang.textContent = lang;
  if (grpStatus) {
    grpStatus.textContent = isAnalyzed ? 'Analyzed ✓' : 'Not Analyzed';
    grpStatus.className = isAnalyzed ? 'badge badge-green' : 'badge badge-amber';
  }
  if (grpEmpty && grpContent) {
    grpEmpty.style.display = isAnalyzed ? 'none' : 'block';
    grpContent.style.display = isAnalyzed ? 'block' : 'none';
  }

  // 4. Update MLIR view banner & empty state
  const mlirTitle = document.getElementById('mlir-active-title');
  const mlirLang = document.getElementById('mlir-active-lang');
  const mlirStatus = document.getElementById('mlir-active-status');
  const mlirEmpty = document.getElementById('mlir-empty-state');
  const mlirContent = document.getElementById('mlir-main-content');

  if (mlirTitle) mlirTitle.textContent = title;
  if (mlirLang) mlirLang.textContent = lang;
  if (mlirStatus) {
    mlirStatus.textContent = isAnalyzed ? 'Analyzed ✓' : 'Not Analyzed';
    mlirStatus.className = isAnalyzed ? 'badge badge-green' : 'badge badge-amber';
  }
  if (mlirEmpty && mlirContent) {
    mlirEmpty.style.display = isAnalyzed ? 'none' : 'block';
    mlirContent.style.display = isAnalyzed ? 'block' : 'none';
  }

  // 5. Update MAX Graph view banner & empty state
  const maxTitle = document.getElementById('max-active-title');
  const maxLang = document.getElementById('max-active-lang');
  const maxStatus = document.getElementById('max-active-status');
  const maxEmpty = document.getElementById('max-empty-state');
  const maxContent = document.getElementById('max-main-content');

  if (maxTitle) maxTitle.textContent = title;
  if (maxLang) maxLang.textContent = lang;
  if (maxStatus) {
    maxStatus.textContent = isAnalyzed ? 'Analyzed ✓' : 'Not Analyzed';
    maxStatus.className = isAnalyzed ? 'badge badge-green' : 'badge badge-amber';
  }
  if (maxEmpty && maxContent) {
    maxEmpty.style.display = isAnalyzed ? 'none' : 'block';
    maxContent.style.display = isAnalyzed ? 'block' : 'none';
  }

  // 6. Update Benchmark view banner & empty state
  const benchTitle = document.getElementById('benchmark-active-title');
  const benchLang = document.getElementById('benchmark-active-lang');
  const benchStatus = document.getElementById('benchmark-active-status');
  const benchEmpty = document.getElementById('benchmark-empty-state');
  const benchContent = document.getElementById('benchmark-main-content');

  if (benchTitle) benchTitle.textContent = title;
  if (benchLang) benchLang.textContent = lang;
  if (benchStatus) {
    benchStatus.textContent = isAnalyzed ? 'Analyzed ✓' : 'Not Analyzed';
    benchStatus.className = isAnalyzed ? 'badge badge-green' : 'badge badge-amber';
  }
  if (benchEmpty && benchContent) {
    benchEmpty.style.display = isAnalyzed ? 'none' : 'block';
    benchContent.style.display = isAnalyzed ? 'block' : 'none';
  }
};

/**
 * Single-Page Application View Routing
 */
function initViewRouter() {
  const navItems = document.querySelectorAll('.sidebar-nav .nav-item');
  const viewSections = document.querySelectorAll('.view-section');
  const crumbTitle = document.getElementById('header-crumb-title');

  window.navigateToView = function(viewName) {
    if (!viewName) return;
    currentActiveView = viewName;

    // Update Nav items active state
    navItems.forEach(item => {
      if (item.dataset.view === viewName) {
        item.classList.add('active');
      } else {
        item.classList.remove('active');
      }
    });

    // Update Section Visibility
    viewSections.forEach(sec => {
      if (sec.id === `view-${viewName}`) {
        sec.classList.add('active');
      } else {
        sec.classList.remove('active');
      }
    });

    // Update Breadcrumb Text
    if (crumbTitle) {
      const activeNav = document.querySelector(`.sidebar-nav .nav-item[data-view="${viewName}"]`);
      if (activeNav) {
        crumbTitle.textContent = activeNav.querySelector('.nav-label')?.textContent || viewName.toUpperCase();
      }
    }

    // Scroll to top
    const mainWrapper = document.querySelector('.main-wrapper');
    if (mainWrapper) mainWrapper.scrollTo({ top: 0, behavior: 'smooth' });

    // Close mobile drawer if open
    const sidebar = document.getElementById('app-sidebar');
    if (sidebar) sidebar.classList.remove('mobile-open');

    // Sync active program banners and empty states
    window.updateActiveProgramUI();

    // Trigger view-specific refreshes
    if (viewName === 'graph' && typeof window.renderGraph === 'function') {
      window.renderGraph();
    } else if (viewName === 'mlir' && typeof window.renderMlirFromContext === 'function') {
      window.renderMlirFromContext();
    } else if (viewName === 'max-graph' && typeof window.renderMaxGraphFromContext === 'function') {
      window.renderMaxGraphFromContext();
    } else if (viewName === 'history' && typeof window.loadSessionHistory === 'function') {
      window.loadSessionHistory();
    }
  };

  // Attach click listeners to sidebar nav items
  navItems.forEach(item => {
    item.addEventListener('click', () => {
      window.navigateToView(item.dataset.view);
    });
  });

  // Handle hash change for deep links
  window.addEventListener('hashchange', () => {
    const hash = window.location.hash.replace('#', '');
    if (hash) {
      window.navigateToView(hash);
    }
  });

  // Check initial hash
  const initialHash = window.location.hash.replace('#', '');
  if (initialHash) {
    window.navigateToView(initialHash);
  }
}

/**
 * Sidebar Collapse & Mobile Drawer Toggle
 */
function initSidebarControls() {
  const sidebar = document.getElementById('app-sidebar');
  const btnToggle = document.getElementById('btn-toggle-sidebar');
  const btnMobile = document.getElementById('btn-mobile-menu');

  if (btnToggle && sidebar) {
    btnToggle.addEventListener('click', () => {
      sidebar.classList.toggle('collapsed');
      const isCollapsed = sidebar.classList.contains('collapsed');
      btnToggle.innerHTML = isCollapsed ? '<span>▶</span>' : '<span>◀</span>';
    });
  }

  if (btnMobile && sidebar) {
    btnMobile.addEventListener('click', () => {
      sidebar.classList.toggle('mobile-open');
    });
  }
}

/**
 * Health Check & Environment Discovery
 */
async function checkSystemStatus() {
  const statusDot = document.getElementById('status-pulse-dot');
  const statusText = document.getElementById('status-pulse-text');
  const runtimeBadge = document.getElementById('header-runtime-badge');
  const runtimeDesc = document.getElementById('header-runtime-desc');
  const mojoStatusBadge = document.getElementById('matrix-mojo-status');

  try {
    const res = await fetch('/api/environment');
    if (res.ok) {
      hostEnvironment = await res.json();
      
      if (statusDot) {
        statusDot.classList.remove('offline');
      }
      if (statusText) {
        statusText.textContent = 'Backend Connected';
      }
      if (runtimeBadge) {
        runtimeBadge.textContent = '● Ready';
      }
      if (runtimeDesc && hostEnvironment) {
        const cores = hostEnvironment.logical_cores || 1;
        const pyVer = hostEnvironment.python_version || '3.12';
        runtimeDesc.textContent = `Python ${pyVer} • ${cores} Cores • AVX2 SIMD`;
      }
      if (mojoStatusBadge && hostEnvironment) {
        if (hostEnvironment.mojo_installed) {
          mojoStatusBadge.textContent = 'Native Live';
          mojoStatusBadge.className = 'badge badge-green';
        } else {
          mojoStatusBadge.textContent = 'AST Simulator';
          mojoStatusBadge.className = 'badge badge-cyan';
        }
      }
    } else {
      throw new Error('Non-200 response');
    }
  } catch (err) {
    console.warn('Backend connection status:', err);
    if (statusDot) statusDot.classList.add('offline');
    if (statusText) statusText.textContent = 'Offline / Reconnecting';
    if (runtimeBadge) {
      runtimeBadge.textContent = '⚠ Offline';
      runtimeBadge.className = 'badge badge-amber';
    }
  }
}

/**
 * Help Onboarding Modal Controller
 */
function initHelpModal() {
  const modal = document.getElementById('modal-help');
  const btnOpen = document.getElementById('btn-open-help');
  const btnClose = document.getElementById('btn-close-help-modal');
  const btnAction = document.getElementById('btn-close-help-action');

  const openModal = () => {
    if (modal) modal.classList.add('active');
  };
  const closeModal = () => {
    if (modal) modal.classList.remove('active');
  };

  if (btnOpen) btnOpen.addEventListener('click', openModal);
  if (btnClose) btnClose.addEventListener('click', closeModal);
  if (btnAction) btnAction.addEventListener('click', () => {
    closeModal();
    window.navigateToView('playground');
  });

  if (modal) {
    modal.addEventListener('click', (e) => {
      if (e.target === modal) closeModal();
    });
  }

  // Escape key closes modals
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      closeModal();
      const optModal = document.getElementById('modal-view-example');
      if (optModal) optModal.classList.remove('active');
    }
  });
}

/**
 * Dashboard Hero Buttons & Workflow Step Click Handlers
 */
function initDashboardActions() {
  // Start Playground
  const btnStart = document.getElementById('btn-dash-start-playground');
  if (btnStart) {
    btnStart.addEventListener('click', () => {
      window.navigateToView('playground');
    });
  }

  // Explore Example
  const btnExplore = document.getElementById('btn-dash-explore-example');
  if (btnExplore) {
    btnExplore.addEventListener('click', () => {
      window.navigateToView('playground');
      if (typeof window.loadPlaygroundExample === 'function') {
        window.loadPlaygroundExample('vec_add');
      }
    });
  }

  // 5-Step Workflow Cards
  document.querySelectorAll('.workflow-step-card').forEach(card => {
    card.addEventListener('click', () => {
      const target = card.dataset.stepTarget || 'playground';
      window.navigateToView(target);
    });
  });

  // Workload Quick Launch
  document.querySelectorAll('.workload-item').forEach(item => {
    item.addEventListener('click', () => {
      const expId = item.dataset.loadExp;
      if (expId) {
        window.navigateToView('playground');
        if (typeof window.loadPlaygroundExample === 'function') {
          window.loadPlaygroundExample(expId);
        }
      }
    });
  });
}
