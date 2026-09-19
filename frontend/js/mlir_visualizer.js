/**
 * AI Compiler & Kernel Playground - Dynamic MLIR Pipeline Visualizer Controller
 * Single Source of Truth: Renders MLIR Dialect Lowering directly from ActiveProgramContext
 */

let activeStageIdx = 0;
let currentMlirMode = 'pipeline'; // 'pipeline', 'original', 'optimized'

document.addEventListener('DOMContentLoaded', () => {
  initMlirControls();
  renderMlirFromContext();
});

function initMlirControls() {
  const btnCopy = document.getElementById('btn-copy-mlir-code');
  const btnPipeline = document.getElementById('btn-mlir-view-pipeline');
  const btnOriginal = document.getElementById('btn-mlir-view-original');
  const btnOptimized = document.getElementById('btn-mlir-view-optimized');

  if (btnPipeline) {
    btnPipeline.addEventListener('click', () => {
      currentMlirMode = 'pipeline';
      updateMlirModeButtons();
      renderMlirFromContext();
    });
  }

  if (btnOriginal) {
    btnOriginal.addEventListener('click', () => {
      currentMlirMode = 'original';
      updateMlirModeButtons();
      renderMlirFromContext();
    });
  }

  if (btnOptimized) {
    btnOptimized.addEventListener('click', () => {
      currentMlirMode = 'optimized';
      updateMlirModeButtons();
      renderMlirFromContext();
    });
  }

  if (btnCopy) {
    btnCopy.addEventListener('click', async () => {
      const code = document.getElementById('mlir-code-display')?.textContent || '';
      try {
        await navigator.clipboard.writeText(code);
        btnCopy.textContent = 'Copied!';
        setTimeout(() => { btnCopy.textContent = 'Copy Code'; }, 1500);
      } catch (err) {
        console.warn('Clipboard failed:', err);
      }
    });
  }
}

function updateMlirModeButtons() {
  const btnPipeline = document.getElementById('btn-mlir-view-pipeline');
  const btnOriginal = document.getElementById('btn-mlir-view-original');
  const btnOptimized = document.getElementById('btn-mlir-view-optimized');

  if (btnPipeline) btnPipeline.classList.toggle('active', currentMlirMode === 'pipeline');
  if (btnOriginal) btnOriginal.classList.toggle('active', currentMlirMode === 'original');
  if (btnOptimized) btnOptimized.classList.toggle('active', currentMlirMode === 'optimized');
}

/**
 * Dynamically Renders the MLIR Pipeline from ActiveProgramContext
 */
function renderMlirFromContext() {
  const emptyState = document.getElementById('mlir-empty-state');
  const mainContent = document.getElementById('mlir-main-content');
  const stepper = document.getElementById('mlir-stepper-container');
  const dialectTag = document.getElementById('mlir-dialect-tag');
  const heading = document.getElementById('mlir-stage-heading');
  const codeDisplay = document.getElementById('mlir-code-display');
  const desc = document.getElementById('mlir-stage-description');
  const passCmd = document.getElementById('mlir-pass-command');
  const conceptsList = document.getElementById('mlir-concepts-list');

  const ctx = window.ActiveProgramContext;
  const isAnalyzed = !!(ctx && ctx.analyzed && ctx.analysis && ctx.analysis.mlir);

  if (!isAnalyzed) {
    if (emptyState) emptyState.style.display = 'block';
    if (mainContent) mainContent.style.display = 'none';
    return;
  }

  if (emptyState) emptyState.style.display = 'none';
  if (mainContent) mainContent.style.display = 'block';

  const mlirData = ctx.analysis.mlir;

  if (currentMlirMode === 'original') {
    if (stepper) stepper.style.display = 'none';
    if (dialectTag) dialectTag.textContent = 'High-Level IR';
    if (heading) heading.textContent = 'Original Program MLIR Emitted';
    if (codeDisplay) codeDisplay.textContent = mlirData.original_mlir || '// No MLIR emitted';
    if (desc) desc.textContent = `Direct dialect IR generated from active ${ctx.language.toUpperCase()} script (${ctx.workload || 'Algorithm'}). Represents structured inputs before optimization passes.`;
    if (passCmd) passCmd.textContent = 'mlir-opt --canonicalize input.mlir';
    if (conceptsList) {
      conceptsList.innerHTML = `
        <li>Static Single Assignment (SSA) form</li>
        <li>High-level dialect representation (${ctx.language === 'mojo' ? 'Mojo Dialect' : 'Python AST IR'})</li>
        <li>Memory tensor layout declarations</li>
      `;
    }
    return;
  }

  if (currentMlirMode === 'optimized') {
    if (stepper) stepper.style.display = 'none';
    if (dialectTag) dialectTag.textContent = 'Vector / LLVM';
    if (heading) heading.textContent = 'Fully Lowered Optimized MLIR';
    if (codeDisplay) codeDisplay.textContent = mlirData.optimized_mlir || '// No optimized MLIR available';
    if (desc) desc.textContent = 'Fully vectorized and lowered MLIR representation ready for target LLVM code generation with hardware SIMD register binding.';
    if (passCmd) passCmd.textContent = 'mlir-opt --convert-vector-to-llvm --convert-func-to-llvm | mlir-translate -mlir-to-llvmir';
    if (conceptsList) {
      conceptsList.innerHTML = `
        <li>Direct 8-wide AVX2 hardware vector registers</li>
        <li>Zero-cost register accumulator</li>
        <li>Direct memory pointer dereference</li>
      `;
    }
    return;
  }

  // Pipeline 5-Stage Mode
  if (stepper) stepper.style.display = 'flex';
  const stages = mlirData.stages || [];
  if (activeStageIdx >= stages.length) activeStageIdx = 0;

  stepper.innerHTML = stages.map((stage, idx) => `
    <div class="pipeline-step-item ${idx === activeStageIdx ? 'active' : ''}" onclick="selectMlirStage(${idx})">
      <div class="pipeline-step-dialect">${stage.dialect || `Stage ${idx + 1}`}</div>
      <div class="pipeline-step-title">${stage.title || `Pass ${idx + 1}`}</div>
    </div>
  `).join('');

  const stage = stages[activeStageIdx] || stages[0];
  if (stage) {
    if (dialectTag) dialectTag.textContent = stage.dialect || 'Dialect';
    if (heading) heading.textContent = stage.title || 'Lowering Stage';
    if (codeDisplay) codeDisplay.textContent = stage.code || '// No IR snippet available';
    if (desc) desc.textContent = stage.description || '';

    if (passCmd) {
      if (stage.stage_id === 'source') {
        passCmd.textContent = 'mlir-opt --emit-source-ir input.py';
      } else if (stage.stage_id === 'linalg') {
        passCmd.textContent = 'mlir-opt --linalg-tile="tile-sizes=64" input.mlir';
      } else if (stage.stage_id === 'scf') {
        passCmd.textContent = 'mlir-opt --convert-linalg-to-loops input.mlir';
      } else if (stage.stage_id === 'vector') {
        passCmd.textContent = 'mlir-opt --affine-vectorize="virtual-vector-size=8" input.mlir';
      } else if (stage.stage_id === 'llvm') {
        passCmd.textContent = 'mlir-opt --convert-vector-to-llvm --convert-func-to-llvm input.mlir';
      } else {
        passCmd.textContent = 'mlir-opt --canonicalize input.mlir';
      }
    }

    if (conceptsList) {
      const concepts = stage.key_concepts || ['Compiler Intermediate Representation', 'Progressive Lowering'];
      conceptsList.innerHTML = concepts.map(c => `<li>${c}</li>`).join('');
    }
  }
}

window.selectMlirStage = function(stageIdx) {
  activeStageIdx = stageIdx;
  currentMlirMode = 'pipeline';
  updateMlirModeButtons();
  renderMlirFromContext();
};

window.renderMlirFromContext = renderMlirFromContext;

