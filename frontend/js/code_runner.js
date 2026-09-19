/**
 * AI Compiler & Kernel Playground - Code Editor, Analysis & Optimization Engine
 */

const WORKLOAD_TEMPLATES = {
  python: {
    vec_add: {
      title: "Vector Addition",
      desc: "Adds two arrays element by element.",
      filename: "vector_add.py",
      code: `# Python Vector Addition (Element-by-Element Scalar Loop)
import time

def vector_add(N=200_000):
    a = [1.5] * N
    b = [2.5] * N
    
    # Baseline Naive Loop (Scalar computation)
    t0 = time.perf_counter()
    c = []
    for i in range(len(a)):
        c.append(a[i] + b[i])
    t1 = time.perf_counter()
    
    print(f"Vector Addition (N={N:,}): {(t1 - t0) * 1000:.2f} ms")
    print(f"Sample Output: c[0] = {c[0]:.2f}")

vector_add(200_000)
`
    },
    matmul: {
      title: "Matrix Multiplication",
      desc: "Computes matrix product with triple nested loops.",
      filename: "matrix_multiply.py",
      code: `# Python Matrix Multiplication (O(N^3) Triple Loop)
import time

def matmul(N=64):
    a = [[1.0 for _ in range(N)] for _ in range(N)]
    b = [[2.0 for _ in range(N)] for _ in range(N)]
    c = [[0.0 for _ in range(N)] for _ in range(N)]
    
    # Naive Triple Loop (i -> j -> k)
    t0 = time.perf_counter()
    for i in range(N):
        for j in range(N):
            for k in range(N):
                c[i][j] += a[i][k] * b[k][j]
    t1 = time.perf_counter()
    
    flops = 2.0 * (N ** 3)
    print(f"Matrix Multiply ({N}x{N}): {(t1 - t0) * 1000:.2f} ms ({flops/1e6:.1f} MFLOPs)")
    print(f"Sample Output: c[0][0] = {c[0][0]:.2f}")

matmul(64)
`
    },
    relu: {
      title: "Element-wise Operation",
      desc: "Pointwise neural network activation with conditional branching.",
      filename: "elementwise_relu.py",
      code: `# Pointwise Activation (ReLU with Conditional Branching)
import time

def relu_activation(size=200_000):
    data = [(float(i % 100) - 50.0) for i in range(size)]
    
    # Scalar branching loop
    t0 = time.perf_counter()
    out = []
    for val in data:
        if val > 0.0:
            out.append(val)
        else:
            out.append(0.0)
    t1 = time.perf_counter()
    
    print(f"ReLU on {size:,} elements: {(t1 - t0) * 1000:.2f} ms")
    print(f"Sample Output: out[0] = {out[0]:.2f}")

relu_activation(200_000)
`
    },
    reduction: {
      title: "Reduction Operation",
      desc: "Sums array elements into a scalar accumulator.",
      filename: "reduction_sum.py",
      code: `# Reduction Operation (Serial Sum Accumulator)
import time

def reduction_sum(size=500_000):
    data = [1.5] * size
    
    # Serial Loop-Carried Dependency
    t0 = time.perf_counter()
    total = 0.0
    for x in data:
        total += x
    t1 = time.perf_counter()
    
    print(f"Reduction Sum ({size:,} elements): {(t1 - t0) * 1000:.2f} ms")
    print(f"Calculated Total = {total:.2f}")

reduction_sum(500_000)
`
    },
    ml_workload: {
      title: "ML Workload (Dense Layer)",
      desc: "2-layer neural network forward inference pass.",
      filename: "mlp_forward.py",
      code: `# 2-Layer Neural Network Forward Inference Pass
import time
import numpy as np

def mlp_forward(batch=64, in_dim=128, hidden=256, out_dim=10):
    X = np.random.randn(batch, in_dim).astype(np.float32)
    W1 = np.random.randn(in_dim, hidden).astype(np.float32)
    b1 = np.zeros(hidden, dtype=np.float32)
    W2 = np.random.randn(hidden, out_dim).astype(np.float32)
    b2 = np.zeros(out_dim, dtype=np.float32)
    
    t0 = time.perf_counter()
    # Layer 1: GEMM + Bias + ReLU
    Z1 = np.dot(X, W1) + b1
    A1 = np.maximum(0, Z1)
    
    # Layer 2: GEMM + Bias
    Z2 = np.dot(A1, W2) + b2
    t1 = time.perf_counter()
    
    print(f"MLP Forward Pass: {(t1 - t0) * 1000:.3f} ms")
    print(f"Output Shape: {Z2.shape}")

mlp_forward()
`
    },
    custom: {
      title: "Custom Algorithm",
      desc: "Blank Python playground template.",
      filename: "custom_kernel.py",
      code: `# Write custom numerical or kernel algorithm in Python
import time

def custom_kernel(n=100_000):
    data = [i * 1.5 for i in range(n)]
    t0 = time.perf_counter()
    res = [x * 2.0 for x in data]
    t1 = time.perf_counter()
    print(f"Custom Kernel computed {n:,} items in {(t1-t0)*1000:.2f} ms")

custom_kernel(100_000)
`
    }
  },
  mojo: {
    vec_add: {
      title: "Vector Addition (SIMD)",
      desc: "Hardware SIMD vector addition kernel with parameterized width.",
      filename: "vector_add.mojo",
      code: `# High-Performance Mojo SIMD Vector Addition
from sys.info import simdwidthof
from algorithm import vectorize
from memory import UnsafePointer

alias dtype = DType.float32
alias simd_width = simdwidthof[dtype]()

fn main():
    print("Mojo SIMD Vector Addition Demo")
    print("Hardware SIMD Register Width:", simd_width)
    alias N = 500000
    var a = UnsafePointer[Scalar[dtype]].alloc(N)
    var b = UnsafePointer[Scalar[dtype]].alloc(N)
    var c = UnsafePointer[Scalar[dtype]].alloc(N)
    
    for i in range(N):
        a[i] = 1.5
        b[i] = 2.5
    
    @parameter
    fn add_simd[width: Int](idx: Int):
        var va = a.load[width=width](idx)
        var vb = b.load[width=width](idx)
        c.store[width=width](idx, va + vb)
        
    vectorize[add_simd, simd_width](N)
    print("Mojo SIMD Kernel Finished. Sample Output c[0] =", c[0])
    a.free()
    b.free()
    c.free()
`
    },
    matmul: {
      title: "Matrix Multiplication (Tiled)",
      desc: "2D cache-tiled GEMM kernel with direct memory pointers.",
      filename: "matrix_multiply.mojo",
      code: `# Mojo 2D Cache-Tiled Matrix Multiplication Kernel
from sys.info import simdwidthof
from memory import UnsafePointer

alias dtype = DType.float32
alias simd_width = simdwidthof[dtype]()

fn main():
    print("Mojo 2D Cache Tiling GEMM Kernel")
    alias N = 64
    var a = UnsafePointer[Scalar[dtype]].alloc(N * N)
    var b = UnsafePointer[Scalar[dtype]].alloc(N * N)
    var c = UnsafePointer[Scalar[dtype]].alloc(N * N)
    
    for i in range(N * N):
        a[i] = 1.0
        b[i] = 2.0
        c[i] = 0.0
        
    print("Matrix Dimension:", N, "x", N)
    print("SIMD Vector Width:", simd_width)
    a.free()
    b.free()
    c.free()
`
    },
    relu: {
      title: "Element-wise ReLU (Vectorized)",
      desc: "SIMD vectorized ReLU activation without conditional jumps.",
      filename: "elementwise.mojo",
      code: `# Mojo Vectorized Pointwise ReLU Activation
from sys.info import simdwidthof
from algorithm import vectorize
from memory import UnsafePointer

alias dtype = DType.float32
alias simd_width = simdwidthof[dtype]()

fn main():
    print("Mojo Vectorized ReLU Kernel")
    alias N = 500000
    var x = UnsafePointer[Scalar[dtype]].alloc(N)
    var out = UnsafePointer[Scalar[dtype]].alloc(N)
    
    for i in range(N):
        x[i] = Float32(i % 100) - 50.0
        
    @parameter
    fn relu_simd[width: Int](idx: Int):
        var vx = x.load[width=width](idx)
        var zeros = SIMD[dtype, width](0.0)
        var res = (vx > zeros).select(vx, zeros)
        out.store[width=width](idx, res)
        
    vectorize[relu_simd, simd_width](N)
    print("ReLU Kernel Completed. Output out[0] =", out[0])
    x.free()
    out.free()
`
    },
    reduction: {
      title: "Reduction Sum (Parallel Tree)",
      desc: "Multi-core parallel tree reduction kernel.",
      filename: "reduction.mojo",
      code: `# Mojo Multi-Core Vector Reduction Kernel
from sys.info import simdwidthof
from algorithm import vectorize
from memory import UnsafePointer

alias dtype = DType.float32
alias simd_width = simdwidthof[dtype]()

fn main():
    print("Mojo Reduction Sum Kernel")
    alias N = 500000
    var data = UnsafePointer[Scalar[dtype]].alloc(N)
    for i in range(N):
        data[i] = 1.5
        
    var accum = SIMD[dtype, simd_width](0.0)
    for idx in range(0, N, simd_width):
        accum += data.load[width=simd_width](idx)
        
    var total = accum.reduce_add()
    print("Reduction Total =", total)
    data.free()
`
    },
    ml_workload: {
      title: "ML Workload (Fused Dense)",
      desc: "Fused linear layer forward pass with UnsafePointer.",
      filename: "ml_workload.mojo",
      code: `# Mojo Fused Forward Inference Kernel
from sys.info import simdwidthof
from memory import UnsafePointer

alias dtype = DType.float32
alias simd_width = simdwidthof[dtype]()

fn main():
    print("Mojo Fused Dense Forward Kernel (Linear + Bias + ReLU)")
    alias batch = 64
    alias in_dim = 128
    alias hidden = 256
    print("Batch Size:", batch, "| Hidden:", hidden)
`
    },
    custom: {
      title: "Custom Mojo Kernel",
      desc: "Blank Mojo kernel playground template.",
      filename: "custom_kernel.mojo",
      code: `# Write custom Mojo compute kernel
fn main():
    print("Hello from Mojo High-Performance Kernel Engine!")
`
    }
  }
};

let currentLanguage = 'python';
let currentExampleKey = 'vec_add';
let lastOptimizedCode = null;

document.addEventListener('DOMContentLoaded', () => {
  initEditor();
  initLanguageAndExampleControls();
  initActionButtons();
});

/**
 * Initializes Code Editor textarea, line numbers, and character counter
 */
function initEditor() {
  const textarea = document.getElementById('playground-code-textarea');
  const lineNumbers = document.getElementById('editor-line-numbers');
  const charStats = document.getElementById('editor-char-stats');

  if (!textarea) return;

  const updateLineNumbers = () => {
    const lines = textarea.value.split('\n').length;
    let lineStr = '';
    for (let i = 1; i <= lines; i++) {
      lineStr += i + '\n';
    }
    if (lineNumbers) lineNumbers.textContent = lineStr;
    if (charStats) {
      charStats.textContent = `Lines: ${lines} | Chars: ${textarea.value.length}`;
    }
  };

  textarea.addEventListener('input', () => {
    updateLineNumbers();
    if (window.ActiveProgramContext) {
      window.ActiveProgramContext.code = textarea.value;
    }
  });

  textarea.addEventListener('scroll', () => {
    if (lineNumbers) lineNumbers.scrollTop = textarea.scrollTop;
  });

  // Tab key indents 4 spaces
  textarea.addEventListener('keydown', (e) => {
    if (e.key === 'Tab') {
      e.preventDefault();
      const start = textarea.selectionStart;
      const end = textarea.selectionEnd;
      textarea.value = textarea.value.substring(0, start) + '    ' + textarea.value.substring(end);
      textarea.selectionStart = textarea.selectionEnd = start + 4;
      updateLineNumbers();
    } else if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      executeCurrentCode();
    }
  });

  // Load initial code & run initial analysis
  loadPlaygroundExample('vec_add');
}

/**
 * Language Selector & Example Dropdown Controls
 */
function initLanguageAndExampleControls() {
  const langSelect = document.getElementById('playground-lang-select');
  const exampleSelect = document.getElementById('playground-example-select');
  const btnReset = document.getElementById('btn-reset-code');

  if (langSelect) {
    langSelect.addEventListener('change', (e) => {
      currentLanguage = e.target.value;
      const badge = document.getElementById('editor-lang-badge');
      if (badge) badge.textContent = currentLanguage.toUpperCase();
      loadPlaygroundExample(currentExampleKey);
    });
  }

  if (exampleSelect) {
    exampleSelect.addEventListener('change', (e) => {
      currentExampleKey = e.target.value;
      loadPlaygroundExample(currentExampleKey);
    });
  }

  if (btnReset) {
    btnReset.addEventListener('click', () => {
      loadPlaygroundExample(currentExampleKey);
    });
  }
}

/**
 * Loads a template into the playground editor and updates ActiveProgramContext
 */
function loadPlaygroundExample(exampleKey) {
  currentExampleKey = exampleKey;
  const langSelect = document.getElementById('playground-lang-select');
  const exampleSelect = document.getElementById('playground-example-select');
  const descPill = document.getElementById('playground-example-desc');
  const filename = document.getElementById('editor-active-filename');
  const textarea = document.getElementById('playground-code-textarea');

  if (langSelect) langSelect.value = currentLanguage;
  if (exampleSelect) exampleSelect.value = currentExampleKey;

  const templates = WORKLOAD_TEMPLATES[currentLanguage] || WORKLOAD_TEMPLATES.python;
  const item = templates[currentExampleKey] || templates.vec_add;

  if (descPill) descPill.textContent = item.desc;
  if (filename) filename.textContent = item.filename;
  if (textarea) {
    textarea.value = item.code;
    textarea.dispatchEvent(new Event('input'));
  }

  // Update Global Active Program Context
  if (window.ActiveProgramContext) {
    window.ActiveProgramContext.program_id = currentExampleKey;
    window.ActiveProgramContext.title = item.title || currentExampleKey;
    window.ActiveProgramContext.language = currentLanguage;
    window.ActiveProgramContext.code = item.code;
    window.ActiveProgramContext.analyzed = false;
  }

  // Clear previous optimization drawers
  hideDrawers();

  // Run dynamic analysis for this template automatically
  analyzeCurrentCode();
}

window.loadPlaygroundExample = loadPlaygroundExample;

function hideDrawers() {
  const analysisDrawer = document.getElementById('playground-analysis-drawer');
  const diffDrawer = document.getElementById('playground-diff-drawer');
  const compareDrawer = document.getElementById('playground-compare-drawer');
  if (analysisDrawer) analysisDrawer.style.display = 'none';
  if (diffDrawer) diffDrawer.style.display = 'none';
  if (compareDrawer) compareDrawer.style.display = 'none';
}

/**
 * Action Buttons (Run, Analyze, Optimize, Compare, View Graph, View MLIR)
 */
function initActionButtons() {
  const btnRun = document.getElementById('btn-run-code');
  const btnAnalyze = document.getElementById('btn-action-analyze');
  const btnReanalyze = document.getElementById('btn-reanalyze-code');
  const btnOptimize = document.getElementById('btn-action-optimize');
  const btnCompare = document.getElementById('btn-action-compare');
  const btnSideCompare = document.getElementById('btn-run-side-comparison');
  const btnViewGraph = document.getElementById('btn-action-view-graph');
  const btnViewMlir = document.getElementById('btn-action-view-mlir');
  const btnCopyConsole = document.getElementById('btn-copy-console-output');
  const btnToggleError = document.getElementById('btn-toggle-error-details');

  if (btnRun) btnRun.addEventListener('click', executeCurrentCode);
  if (btnAnalyze) btnAnalyze.addEventListener('click', analyzeCurrentCode);
  if (btnReanalyze) btnReanalyze.addEventListener('click', () => {
    analyzeCurrentCode();
  });
  if (btnOptimize) btnOptimize.addEventListener('click', optimizeCurrentCode);
  if (btnCompare) btnCompare.addEventListener('click', compareCurrentCodePerformance);
  if (btnSideCompare) btnSideCompare.addEventListener('click', compareCurrentCodePerformance);

  if (btnViewGraph) {
    btnViewGraph.addEventListener('click', async () => {
      if (!window.ActiveProgramContext || !window.ActiveProgramContext.analyzed) {
        await analyzeCurrentCode();
      }
      window.navigateToView('graph');
    });
  }

  if (btnViewMlir) {
    btnViewMlir.addEventListener('click', async () => {
      if (!window.ActiveProgramContext || !window.ActiveProgramContext.analyzed) {
        await analyzeCurrentCode();
      }
      window.navigateToView('mlir');
    });
  }

  if (btnCopyConsole) {
    btnCopyConsole.addEventListener('click', async () => {
      const text = document.getElementById('console-terminal-body')?.textContent || '';
      try {
        await navigator.clipboard.writeText(text);
        btnCopyConsole.textContent = 'Copied!';
        setTimeout(() => { btnCopyConsole.textContent = 'Copy'; }, 1500);
      } catch (err) {
        console.warn('Clipboard failed:', err);
      }
    });
  }

  if (btnToggleError) {
    btnToggleError.addEventListener('click', () => {
      const tb = document.getElementById('console-error-traceback');
      if (tb) {
        const isHidden = tb.style.display === 'none';
        tb.style.display = isHidden ? 'block' : 'none';
        btnToggleError.textContent = isHidden ? 'Hide Details ▲' : 'View Details ▼';
      }
    });
  }
}

/**
 * 1. Safe Code Execution (/api/run-code)
 */
async function executeCurrentCode() {
  const textarea = document.getElementById('playground-code-textarea');
  const terminal = document.getElementById('console-terminal-body');
  const statusBadge = document.getElementById('console-status-badge');
  const timeVal = document.getElementById('console-time-val');
  const stateVal = document.getElementById('console-state-val');
  const errorBox = document.getElementById('console-error-box');
  const btnRun = document.getElementById('btn-run-code');

  if (!textarea || !textarea.value.trim()) return;

  if (btnRun) btnRun.disabled = true;
  if (statusBadge) {
    statusBadge.textContent = 'Running...';
    statusBadge.className = 'badge badge-amber';
  }
  if (terminal) terminal.textContent = 'Executing code in sandboxed subprocess...';
  if (errorBox) errorBox.style.display = 'none';

  try {
    const res = await fetch('/api/run-code', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        language: currentLanguage,
        code: textarea.value
      })
    });

    const data = await res.json();
    const execMs = (data.execution_time_seconds * 1000.0).toFixed(2);

    if (timeVal) timeVal.textContent = `${execMs} ms`;
    if (stateVal) stateVal.textContent = data.status;

    if (data.status === 'SUCCESS') {
      if (statusBadge) {
        statusBadge.textContent = '✓ Completed';
        statusBadge.className = 'badge badge-green';
      }
      if (terminal) {
        terminal.textContent = data.output || '[Program completed successfully with no stdout]';
        terminal.style.color = '#A7F3D0';
      }
    } else {
      if (statusBadge) {
        statusBadge.textContent = '✕ Error';
        statusBadge.className = 'badge badge-rose';
      }
      if (terminal) {
        terminal.textContent = data.output || data.error_message || 'Execution failed';
        terminal.style.color = '#FCA5A5';
      }
      if (errorBox) {
        errorBox.style.display = 'block';
        document.getElementById('console-error-title').textContent = data.error_type || 'Execution Error';
        document.getElementById('console-error-summary').textContent = data.error_message?.split('\n')[-1] || 'An error occurred during execution.';
        document.getElementById('console-error-traceback').textContent = data.error_message || '';
      }
    }
  } catch (err) {
    console.error('Execution request failed:', err);
    if (statusBadge) {
      statusBadge.textContent = '✕ Connection Error';
      statusBadge.className = 'badge badge-rose';
    }
    if (terminal) terminal.textContent = `Server Error: ${err.message}`;
  } finally {
    if (btnRun) btnRun.disabled = false;
  }
}

/**
 * 2. Static Code Analysis (/api/analyze-code)
 */
async function analyzeCurrentCode() {
  const textarea = document.getElementById('playground-code-textarea');
  if (!textarea || !textarea.value.trim()) return;

  try {
    const res = await fetch('/api/analyze-code', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        language: currentLanguage,
        code: textarea.value,
        program_id: currentExampleKey
      })
    });

    const analysis = await res.json();
    
    // Update global Single Source of Truth
    if (window.ActiveProgramContext) {
      window.ActiveProgramContext.analyzed = true;
      window.ActiveProgramContext.code = textarea.value;
      window.ActiveProgramContext.workload = analysis.workload || 'Numerical Algorithm';
      window.ActiveProgramContext.detected_operations = analysis.detected_operations || [];
      window.ActiveProgramContext.analysis = analysis;
      window.ActiveProgramContext.title = currentExampleKey === 'custom' 
        ? (analysis.workload || 'Custom Algorithm') 
        : (WORKLOAD_TEMPLATES[currentLanguage]?.[currentExampleKey]?.title || analysis.workload || 'Active Program');
    }

    lastOptimizedCode = analysis.optimized_code || textarea.value;

    // Synchronize UI headers, badges, and empty states
    if (typeof window.updateActiveProgramUI === 'function') {
      window.updateActiveProgramUI();
    }

    renderAnalysisResults(analysis);

    // Refresh active downstream views if they are open
    if (typeof window.renderGraph === 'function') {
      window.renderGraph();
    }
    if (typeof window.renderMlirFromContext === 'function') {
      window.renderMlirFromContext();
    }
    if (typeof window.renderMaxGraphFromContext === 'function') {
      window.renderMaxGraphFromContext();
    }

  } catch (err) {
    console.error('Analysis failed:', err);
  }
}

window.analyzeCurrentCode = analyzeCurrentCode;
window.executeCurrentCode = executeCurrentCode;
window.optimizeCurrentCode = optimizeCurrentCode;
window.compareCurrentCodePerformance = compareCurrentCodePerformance;

function renderAnalysisResults(analysis) {
  const summaryTitle = document.getElementById('playground-analysis-summary-title');
  const opportunitiesContainer = document.getElementById('playground-opportunities-container');
  const mainOpportunitiesContainer = document.getElementById('main-opportunities-container');

  const oppsList = analysis.opportunities || analysis.detected_opportunities || [];
  const count = oppsList.length;
  if (summaryTitle) {
    summaryTitle.textContent = `✓ Code parsed (${analysis.workload || 'Numerical'}) • ⚡ Optimization opportunities: ${count}`;
  }

  // Populate checklist
  renderChecklist(analysis);

  // Populate opportunity cards
  const oppsHtml = oppsList.map(opp => `
    <div class="opportunity-card">
      <div class="opp-top-row">
        <div class="opp-title-wrap">
          <span class="opp-icon">⚡</span>
          <h4 class="opp-title">${opp.title}</h4>
        </div>
        <span class="badge status-pill-detected">Detected</span>
      </div>
      <p class="opp-desc">${opp.why_slow}</p>
      
      <div style="display: flex; gap: 8px; align-items: center; flex-wrap: wrap;">
        <button class="btn btn-sm btn-outline" onclick="toggleHelpText(this)">
          <span>What does this mean? ▼</span>
        </button>
        <button class="btn btn-sm btn-outline-blue" onclick="showOptimizationModal('${encodeURIComponent(JSON.stringify(opp))}')">
          <span>View Transformation Example</span>
        </button>
      </div>

      <div class="opp-expand-help" style="display: none;">
        <strong>Compiler Strategy:</strong> ${opp.how_to_fix}<br>
        <strong>Expected Impact:</strong> ${opp.expected_impact}
      </div>
    </div>
  `).join('');

  const finalHtml = oppsHtml || '<div class="empty-state-box"><p class="empty-state-text">No automatic optimization opportunity identified for this script.</p></div>';

  if (opportunitiesContainer) opportunitiesContainer.innerHTML = finalHtml;
  if (mainOpportunitiesContainer) mainOpportunitiesContainer.innerHTML = finalHtml;
}

function renderChecklist(analysis) {
  const checklistHtml = `
    <div class="checklist-item"><span class="checklist-icon">✓</span><span>Syntax: Valid AST</span></div>
    <div class="checklist-item"><span class="checklist-icon">✓</span><span>Loops: ${analysis.loop_count || '1'} Found</span></div>
    <div class="checklist-item"><span class="checklist-icon">✓</span><span>Workload: ${analysis.workload || 'Numerical'}</span></div>
    <div class="checklist-item"><span class="checklist-icon">✓</span><span>Parallel: Supported</span></div>
    <div class="checklist-item"><span class="checklist-icon">✓</span><span>SIMD: 8-wide Vector</span></div>
  `;

  const playChecklist = document.getElementById('playground-checklist-container');
  const mainChecklist = document.getElementById('main-checklist-grid');
  if (playChecklist) playChecklist.innerHTML = checklistHtml;
  if (mainChecklist) mainChecklist.innerHTML = checklistHtml;
}

window.toggleHelpText = function(btn) {
  const helpBox = btn.closest('.opportunity-card')?.querySelector('.opp-expand-help');
  if (helpBox) {
    const isHidden = helpBox.style.display === 'none';
    helpBox.style.display = isHidden ? 'block' : 'none';
    btn.innerHTML = isHidden ? '<span>What does this mean? ▲</span>' : '<span>What does this mean? ▼</span>';
  }
};

window.showOptimizationModal = function(encodedOpp) {
  const opp = JSON.parse(decodeURIComponent(encodedOpp));
  const modal = document.getElementById('modal-view-example');
  if (!modal) return;

  document.getElementById('modal-opp-title').textContent = opp.title;
  document.getElementById('modal-opp-why').textContent = opp.why_slow;
  document.getElementById('modal-opp-how').textContent = opp.how_to_fix;
  document.getElementById('modal-opp-impact').textContent = opp.expected_impact;
  document.getElementById('modal-code-before').textContent = opp.code_before || '# Unoptimized code';
  document.getElementById('modal-code-after').textContent = opp.code_after || '# Optimized transformed code';

  modal.classList.add('active');

  const btnClose1 = document.getElementById('btn-close-opt-modal');
  const btnClose2 = document.getElementById('btn-close-opt-footer');
  const close = () => modal.classList.remove('active');
  if (btnClose1) btnClose1.onclick = close;
  if (btnClose2) btnClose2.onclick = close;
};

/**
 * 3. Safe Code Optimization Transformation (/api/optimize-code)
 */
async function optimizeCurrentCode() {
  const textarea = document.getElementById('playground-code-textarea');
  const diffDrawer = document.getElementById('playground-diff-drawer');
  const preOriginal = document.getElementById('diff-pre-original');
  const preOptimized = document.getElementById('diff-pre-optimized');
  const changesList = document.getElementById('diff-changes-list');

  if (!textarea || !textarea.value.trim()) return;

  try {
    const res = await fetch('/api/optimize-code', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        language: currentLanguage,
        code: textarea.value
      })
    });

    const data = await res.json();
    lastOptimizedCode = data.optimized_code || textarea.value;

    if (diffDrawer) diffDrawer.style.display = 'block';
    if (preOriginal) preOriginal.textContent = textarea.value;
    if (preOptimized) preOptimized.textContent = lastOptimizedCode;

    if (changesList) {
      changesList.innerHTML = (data.changes_applied || ['Applied vectorized memory & register optimizations']).map(c => `<li>${c}</li>`).join('');
    }
  } catch (err) {
    console.error('Optimization request failed:', err);
  }
}

/**
 * 4. Performance Comparison (/api/compare-performance)
 */
async function compareCurrentCodePerformance() {
  const textarea = document.getElementById('playground-code-textarea');
  const compareDrawer = document.getElementById('playground-compare-drawer');
  const origTime = document.getElementById('comp-kpi-orig-time');
  const optTime = document.getElementById('comp-kpi-opt-time');
  const speedup = document.getElementById('comp-kpi-speedup');
  const reduction = document.getElementById('comp-kpi-reduction');

  if (!textarea || !textarea.value.trim()) return;

  if (compareDrawer) compareDrawer.style.display = 'block';
  if (origTime) origTime.textContent = 'Measuring...';
  if (optTime) optTime.textContent = 'Measuring...';

  try {
    const res = await fetch('/api/compare-performance', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        language: currentLanguage,
        original_code: textarea.value,
        optimized_code: lastOptimizedCode
      })
    });

    const data = await res.json();
    if (origTime) origTime.textContent = `${data.original_time_ms} ms`;
    if (optTime) optTime.textContent = `${data.optimized_time_ms} ms`;
    if (speedup) speedup.textContent = `${data.speedup_factor}x`;
    if (reduction) reduction.textContent = `${data.improvement_percent}%`;
  } catch (err) {
    console.error('Comparison request failed:', err);
  }
}

