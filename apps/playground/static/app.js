"use strict";

const $ = (id) => document.getElementById(id);
const TYPES = { choice: ["선택", "choice"], score: ["점수", "score"], noul: ["참·거짓", "noul"] };
const PHOTO_QUESTIONS = {
  material: { name: "material", type: "choice", instructions: "What material category best describes the main item in the photograph?", options: [
    { key: "glass", text: "Glass." }, { key: "paper", text: "Paper." },
    { key: "cardboard", text: "Cardboard." }, { key: "plastic", text: "Plastic." },
    { key: "metal", text: "Metal." }, { key: "other", text: "Other material or not enough visual evidence." },
  ] },
  visibility: { name: "object_visibility", type: "score", instructions: "How clearly can the main object's shape and surface details be inspected? Use the ordered visibility grades; do not score your confidence in another answer.", options: [
    { key: "0", text: "Unusable: the main object cannot be distinguished well enough to inspect its shape." },
    { key: "1", text: "Limited: the object can be located, but most of its shape or surface detail is hidden by blur, darkness, cropping, or occlusion." },
    { key: "2", text: "Usable: the main shape and some surface details are visible, but blur, darkness, cropping, or occlusion prevents a clear view of nearly the whole object." },
    { key: "3", text: "Clear: nearly the whole main object and its surface details are clearly visible, with little blur, darkness, cropping, or occlusion." },
  ] },
  metal: { name: "made_of_metal", type: "noul", instructions: "Is the main object in the photograph primarily made of metal?", options: [
    { key: "false", text: "The main object is not primarily made of metal." },
    { key: "true", text: "The main object is primarily made of metal." },
  ] },
};
const PRESETS = {
  support: {
    text: "Hi, I was charged twice for my subscription this month. Both payments have cleared. Could you check the duplicate charge and help me get a refund?",
    questions: [{ name: "department", type: "choice", instructions: "Which team should handle this customer request?", options: [
      { key: "billing", text: "Billing: payments, invoices, charges and refunds." },
      { key: "technical", text: "Technical support: bugs, login problems and product errors." },
      { key: "sales", text: "Sales: pricing, plans and new purchases." },
      { key: "other", text: "Other: requests unrelated to billing, technical support or sales." },
    ] }],
  },
  policy: {
    text: "Policy: Approve an order only if payment is confirmed AND stock is available. If payment is unconfirmed, hold the order. If payment is confirmed but stock is unavailable, request restocking.\n\nOrder: Payment is confirmed. Stock is unavailable. The customer is a long-term customer.",
    questions: [{ name: "next_action", type: "choice", instructions: "Which action follows the stated order policy?", options: [
      { key: "approve", text: "Approve the order for dispatch." },
      { key: "hold", text: "Hold the order until payment is confirmed." },
      { key: "restock", text: "Request restocking before dispatch." },
    ] }],
  },
  uncertain: {
    text: "A customer says they may have been charged twice. Only one payment receipt is attached. No bank statement or payment processor log is available. The support team has not investigated yet.",
    questions: [{ name: "duplicate_charge", type: "noul", instructions: "Does the available evidence establish that two payments were collected?", options: [
      { key: "false", text: "The available evidence does not establish that two payments were collected." },
      { key: "true", text: "The available evidence establishes that two payments were collected." },
    ] }],
  },
  photo: {
    requiresImage: true,
    sampleId: "sample-01",
    text: "Use the attached photograph as evidence. Identify the material of the main item.",
    questions: [PHOTO_QUESTIONS.material],
  },
  photo_score: {
    requiresImage: true,
    sampleId: "sample-05",
    text: "Use only the attached photograph to rate how well the main object can be inspected. Consider visible shape and surface details, including blur, darkness, cropping, and occlusion. Apply the supplied 0 to 3 visibility rubric.",
    questions: [PHOTO_QUESTIONS.visibility],
  },
  photo_policy: {
    requiresImage: true,
    sampleId: "sample-03",
    text: "Demo sorting policy: Put paper and cardboard in the paper bin. Put glass in the glass bin. Put metal and plastic in the container bin. Send mixed materials or items that cannot be identified confidently for manual review. Apply this stated policy to the main item in the photograph.",
    questions: [{ name: "sorting_action", type: "choice", instructions: "Which action follows the stated demo sorting policy for the photographed item?", options: [
      { key: "paper_bin", text: "Place the item in the paper and cardboard bin." },
      { key: "glass_bin", text: "Place the item in the glass bin." },
      { key: "container_bin", text: "Place the item in the metal and plastic container bin." },
      { key: "manual_review", text: "Send the item for manual review because its material is mixed or unclear." },
    ] }],
  },
  photo_truth: {
    requiresImage: true,
    sampleId: "sample-05",
    text: "Use the attached photograph as the evidence for the proposition. Judge the main object's material, rather than the background or a small label.",
    questions: [PHOTO_QUESTIONS.metal],
  },
  photo_all: {
    requiresImage: true,
    sampleId: "sample-05",
    text: "Use the attached photograph as evidence for all three questions about the main object. Judge its primary material, how clearly its shape and surface details can be inspected, and whether it is primarily made of metal. Apply each question's own criteria using only visible evidence.",
    questions: [PHOTO_QUESTIONS.material, PHOTO_QUESTIONS.visibility, PHOTO_QUESTIONS.metal],
  },
};

const SAMPLE_PHOTOS = [
  { id: "sample-01", label: "갈색 유리병" },
  { id: "sample-02", label: "신문지" },
  { id: "sample-03", label: "골판지" },
  { id: "sample-04", label: "플라스틱 물병" },
  { id: "sample-05", label: "금속 캔" },
  { id: "sample-06", label: "포장 파우치" },
];
const sampleUploads = new Map();

const state = {
  presetKey: "support",
  text: PRESETS.support.text,
  questions: structuredClone(PRESETS.support.questions),
  image: null,
  sourceImage: null,
  resolution: "original",
  imagePolicy: null,
  imageLongEdges: [],
  comparing: false,
  compareStop: false,
  comparison: null,
  phase: "loading",
  running: false,
  uploading: false,
  view: "result",
  last: null,
  history: [],
  errorKind: null,
};
let toastTimer;

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  if (tag === "button") node.type = "button";
  return node;
}

function button(text, className, action) {
  const node = element("button", className, text);
  node.addEventListener("click", action);
  return node;
}

function showError(message = "", kind = "request") {
  $("global-error").textContent = message;
  $("global-error").hidden = !message;
  state.errorKind = message ? kind : null;
}

function toast(message) {
  clearTimeout(toastTimer);
  $("toast").textContent = message;
  $("toast").hidden = false;
  toastTimer = setTimeout(() => { $("toast").hidden = true; }, 3000);
}

function makeRequest(strict = false, image = state.image) {
  if (strict && !state.text.trim() && !image) throw new Error("상황을 입력하거나 사진 한 장을 추가하세요.");
  const usedNames = new Set();
  const entries = state.questions.map((question, index) => {
    const name = question.name.trim() || `question_${index + 1}`;
    if (strict && (!question.name.trim() || usedNames.has(name))) throw new Error("질문 이름은 비워두거나 중복해서 사용할 수 없습니다.");
    usedNames.add(name);
    if (strict && !question.instructions.trim()) throw new Error(`${name}: 질문을 입력하세요.`);
    const result = { type: question.type, instructions: question.instructions };
    if (question.type === "score") result.criteria = question.options.map((option) => option.text);
    else {
      const keys = question.options.map((option) => option.key.trim());
      if (strict && (keys.some((key) => !key) || new Set(keys).size !== keys.length)) throw new Error(`${name}: 후보 ID를 중복 없이 입력하세요.`);
      result.criteria = Object.fromEntries(question.options.map((option, i) => [keys[i], option.text]));
    }
    if (strict && question.options.some((option) => !option.text.trim())) throw new Error(`${name}: 모든 후보의 의미를 입력하세요.`);
    if (strict && question.type === "choice" && new Set(question.options.map((option) => option.text.trim())).size !== question.options.length) throw new Error(`${name}: 후보마다 서로 다른 의미를 입력하세요.`);
    return [name, result];
  });
  return {
    model: "Qwen/Qwen3.5-2B",
    state: { text: state.text, images: image ? [{ path: image.path }] : [] },
    questions: Object.fromEntries(entries),
  };
}

function edited() {
  const request = makeRequest();
  $("request-json").textContent = JSON.stringify(request, null, 2);
  $("char-count").textContent = `${state.text.length.toLocaleString("ko-KR")}자`;
  $("stale-notice").hidden = !state.last || JSON.stringify(request) === JSON.stringify(state.last.request);
  $("comparison-stale").hidden = !state.comparison || comparisonFingerprint() === state.comparison.fingerprint;
}

function updateControls() {
  const locked = state.running || state.uploading;
  $("editor-fields").disabled = locked;
  $("preset").disabled = locked;
  $("run-button").disabled = locked || state.phase !== "ready";
  $("run-label").textContent = state.comparing ? "해상도 비교 중…" : state.running ? "판단 중…" : state.uploading ? "사진 준비 중…" : state.phase === "loading" ? "모델 준비 중" : state.phase !== "ready" ? "연결 대기 중" : "판단 실행";
  $("question-add").disabled = state.questions.length >= 4;
  $("loading-overlay").hidden = !state.running || state.comparing;
  $("resolution-compare").disabled = locked || state.phase !== "ready" || !state.image || !state.imageLongEdges.length;
  $("comparison-cancel").hidden = !state.comparing;
  $("comparison-cancel").disabled = state.compareStop;
  $("comparison-cancel").textContent = state.compareStop ? "현재 요청 완료 후 중단" : "비교 중단";
  $("comparison-export").disabled = state.comparing || !state.comparison?.rows.length;
  document.querySelectorAll("#comparison-rows button").forEach((control) => { control.disabled = locked; });
}

function renderQuestions() {
  const container = $("questions");
  container.replaceChildren();
  state.questions.forEach((question, index) => {
    const card = element("section", "question-card");
    const heading = element("div", "question-heading");
    const nameLabel = element("label", "field-label", "이름");
    nameLabel.htmlFor = `question-name-${index}`;
    const name = element("input", "question-name");
    name.type = "text";
    name.id = nameLabel.htmlFor;
    name.maxLength = 128;
    name.value = question.name;
    name.autocomplete = "off";
    name.addEventListener("input", () => { question.name = name.value; edited(); });
    const remove = button("삭제", "text-button question-remove", () => {
      state.questions.splice(index, 1);
      renderQuestions();
      edited();
    });
    remove.disabled = state.questions.length === 1;
    remove.setAttribute("aria-label", `${index + 1}번 질문 삭제`);
    heading.append(nameLabel, name, remove);
    const types = element("div", "type-switch");
    types.setAttribute("role", "group");
    types.setAttribute("aria-label", `${index + 1}번 질문 유형`);
    Object.entries(TYPES).forEach(([type, label]) => {
      const control = button(label[0], type === question.type ? "active" : "", () => {
        if (type === question.type) return;
        question.type = type;
        question.options = type === "noul" ? [
          { key: "false", text: "The proposition is false." },
          { key: "true", text: "The proposition is true." },
        ] : question.options.map((option, i) => ({ key: type === "score" ? String(i) : `option_${i + 1}`, text: option.text }));
        renderQuestions();
        edited();
      });
      control.setAttribute("aria-pressed", String(type === question.type));
      control.append(element("small", "", label[1]));
      types.append(control);
    });
    const instructionLabel = element("label", "field-label", "무엇을 판단할까요?");
    instructionLabel.htmlFor = `instructions-${index}`;
    const instructions = element("textarea", "question-instructions");
    instructions.id = instructionLabel.htmlFor;
    instructions.rows = 2;
    instructions.maxLength = 8000;
    instructions.spellcheck = false;
    instructions.value = question.instructions;
    instructions.placeholder = "상황과 사진에 대해 판단할 질문을 입력하세요.";
    instructions.addEventListener("input", () => { question.instructions = instructions.value; edited(); });
    const criteriaHeading = element("div", "criteria-heading");
    criteriaHeading.append(element("span", "field-label", question.type === "score" ? "등급별 기준" : question.type === "noul" ? "참·거짓의 의미" : "후보와 의미"));
    criteriaHeading.append(element("span", "criteria-hint", question.type === "score" ? "위에서부터 0, 1, 2… 등급" : question.type === "noul" ? "출력은 참일 확률입니다" : "ID는 의미와 별도로 지정합니다"));
    const criteria = element("div");
    question.options.forEach((option, optionIndex) => {
      const row = element("div", "criterion");
      if (question.type === "choice") {
        const key = element("input", "criterion-key");
        key.type = "text";
        key.value = option.key;
        key.maxLength = 128;
        key.setAttribute("aria-label", `${index + 1}번 질문 후보 ${optionIndex + 1} ID`);
        key.addEventListener("input", () => { option.key = key.value; edited(); });
        row.append(key);
      } else row.append(element("span", "fixed-key", question.type === "score" ? `등급 ${optionIndex}` : option.key));
      const meaning = element("textarea");
      meaning.rows = 1;
      meaning.maxLength = 8000;
      meaning.value = option.text;
      meaning.spellcheck = false;
      meaning.placeholder = "이 후보가 의미하는 내용을 입력하세요.";
      meaning.setAttribute("aria-label", `${index + 1}번 질문 후보 ${optionIndex + 1} 의미`);
      meaning.addEventListener("input", () => { option.text = meaning.value; edited(); });
      row.append(meaning);
      if (question.type !== "noul") {
        const removeOption = button("×", "remove-criterion", () => {
          question.options.splice(optionIndex, 1);
          renderQuestions();
          edited();
        });
        removeOption.disabled = question.options.length <= 2;
        removeOption.setAttribute("aria-label", `후보 ${optionIndex + 1} 삭제`);
        row.append(removeOption);
      }
      criteria.append(row);
    });
    card.append(heading, types, instructionLabel, instructions, criteriaHeading, criteria);
    if (question.type !== "noul") {
      const addOption = button("+ 후보 추가", "text-button criterion-add", () => {
        let next = question.options.length + 1;
        while (question.options.some((option) => option.key === `option_${next}`)) next += 1;
        question.options.push({ key: `option_${next}`, text: "" });
        renderQuestions();
        edited();
      });
      addOption.disabled = question.options.length >= 16;
      card.append(addOption);
    }
    container.append(card);
  });
  $("question-count").textContent = `${state.questions.length} / 4`;
  updateControls();
}

function renderImage() {
  $("image-preview").hidden = !state.image;
  $("upload-zone").hidden = Boolean(state.image);
  if (state.image) {
    $("image-thumbnail").src = state.image.url;
    $("image-name").textContent = state.image.name;
    $("image-size").textContent = `${state.image.width} × ${state.image.height} · ${(state.image.bytes / 1024 / 1024).toFixed(2)} MB`;
  } else $("image-thumbnail").removeAttribute("src");
  document.querySelectorAll("[data-sample]").forEach((control) => {
    const selected = control.dataset.sample === state.image?.sampleId;
    control.classList.toggle("selected", selected);
    control.setAttribute("aria-pressed", String(selected));
  });
  renderResolution();
}

function imageDimensions(image) { return image ? `${image.width} × ${image.height}` : "—"; }

function imageBytes(bytes) {
  return bytes < 1024 * 1024 ? `${(bytes / 1024).toFixed(1)} KB` : `${(bytes / 1024 / 1024).toFixed(2)} MB`;
}

function attachImage(image, source = image) {
  state.image = image;
  state.sourceImage = image ? source : null;
  state.resolution = image?.long_edge || "original";
}

function renderResolution() {
  const source = state.sourceImage;
  const selected = state.image;
  $("resolution-picker").hidden = !source || !selected;
  if (!source || !selected) return;
  const options = $("resolution-options");
  options.replaceChildren();
  ["original", ...state.imageLongEdges].forEach((edge) => {
    const control = button(edge === "original" ? "원본" : `${edge}px`, "resolution-option", () => selectResolution(edge));
    control.setAttribute("aria-pressed", String(edge === state.resolution));
    control.classList.toggle("selected", edge === state.resolution);
    control.disabled = edge !== "original" && edge >= Math.max(source.width, source.height);
    if (control.disabled) control.title = "원본보다 크게 확대하지 않습니다. 원본 크기를 사용하세요.";
    options.append(control);
  });
  $("source-image-preview").src = source.url;
  $("source-image-link").href = source.url;
  $("selected-image-preview").src = selected.url;
  $("selected-image-link").href = selected.url;
  $("source-image-dimensions").textContent = `${imageDimensions(source)} · ${imageBytes(source.bytes)}`;
  $("selected-image-dimensions").textContent = `${imageDimensions(selected)} · ${imageBytes(selected.bytes)}`;
  const geometry = selected.processing;
  $("resolution-model-size").textContent = geometry
    ? `모델 입력 예상 ${imageDimensions(geometry)} · 사진 토큰 ${geometry.visual_tokens}개 / 질문 1개 기준`
    : "모델 입력 크기는 모델이 준비되면 표시됩니다.";
  const policy = state.imagePolicy;
  const notices = [];
  if (policy) {
    notices.push(`현재 모델은 ${policy.min_pixels.toLocaleString("ko-KR")}~${policy.max_pixels.toLocaleString("ko-KR")} 픽셀 면적과 ${policy.alignment}px 단위에 맞춰 사진 크기를 조정합니다.`);
    if (selected.width * selected.height < policy.min_pixels) notices.push("작은 사진은 내부에서 다시 확대됩니다. 잃은 세부 정보는 복원되지 않으며, 추론 시간이 더 줄지 않을 수 있습니다.");
    else if (selected.width * selected.height > policy.max_pixels) notices.push("선택한 사진은 모델 입력 전에 더 작게 축소됩니다.");
  }
  notices.push("축소본은 PNG로 저장해 추가 손실 압축을 피합니다. 원본 JPEG보다 파일이 커질 수 있습니다.");
  $("resolution-hint").textContent = notices.join(" ");
  const count = state.imageLongEdges.filter((edge) => edge < Math.max(source.width, source.height)).length + 1;
  $("resolution-caption").textContent = `${count}개 크기 · 크기별 예열 1회 + 측정 3회 · 사진을 누르면 크게 볼 수 있습니다.`;
}

async function resolutionImage(edge, source = state.sourceImage) {
  if (!source) throw new Error("비교할 사진을 먼저 선택하세요.");
  const response = edge === "original"
    ? await fetch(`/api/images/${source.path}/info`, { cache: "no-store", signal: AbortSignal.timeout(30000) })
    : await fetch(`/api/images/${source.path}/resize`, {
      method: "POST", headers: { "X-Veyra-Playground": "1", "Content-Type": "application/json" },
      body: JSON.stringify({ long_edge: edge }), signal: AbortSignal.timeout(30000),
    });
  return { ...await responseData(response), name: source.name, sampleId: source.sampleId };
}

async function selectResolution(edge) {
  if (state.running || state.uploading || !state.sourceImage) return;
  state.uploading = true;
  updateControls();
  showError();
  try {
    state.image = await resolutionImage(edge);
    state.resolution = state.image.long_edge || "original";
    if (state.resolution === "original") state.sourceImage = state.image;
    renderImage();
    edited();
  } catch (error) { showError(error.message || "해상도를 변경하지 못했습니다."); }
  finally { state.uploading = false; updateControls(); }
}

function renderSamples() {
  const list = $("sample-photos");
  list.replaceChildren();
  SAMPLE_PHOTOS.forEach((sample) => {
    const control = button("", "sample-photo", () => selectSample(sample.id));
    control.dataset.sample = sample.id;
    control.setAttribute("aria-label", `${sample.label} 샘플 사진 삽입`);
    control.setAttribute("aria-pressed", "false");
    const thumbnail = element("img");
    thumbnail.src = `/static/samples/${sample.id}.jpg`;
    thumbnail.alt = "";
    thumbnail.width = 512;
    thumbnail.height = 384;
    thumbnail.loading = "lazy";
    control.append(thumbnail, element("span", "", sample.label));
    list.append(control);
  });
}

function applyPreset(key, image) {
  const preset = PRESETS[key];
  state.presetKey = key;
  state.text = preset.text;
  state.questions = structuredClone(preset.questions);
  attachImage(image, image?.source_path === state.sourceImage?.path ? state.sourceImage : image);
  $("preset").value = key;
  $("state-text").value = state.text;
  renderQuestions();
  renderImage();
  edited();
  showError();
}

async function selectPreset(key) {
  if (state.running || state.uploading) return;
  const preset = PRESETS[key];
  if (preset.requiresImage && !state.image) {
    await selectSample(preset.sampleId, key);
    return;
  }
  applyPreset(key, preset.requiresImage ? state.image : null);
}

async function storeImage(bytes, name) {
  const response = await fetch("/api/images", { method: "POST", headers: { "X-Veyra-Playground": "1", "Content-Type": bytes.type || "application/octet-stream" }, body: bytes, signal: AbortSignal.timeout(30000) });
  return { ...await responseData(response), name };
}

async function selectSample(id, presetKey = null) {
  if (state.running || state.uploading) return;
  const sample = SAMPLE_PHOTOS.find((item) => item.id === id);
  if (!sample) return;
  const nextPreset = presetKey || (PRESETS[state.presetKey]?.requiresImage ? null : "photo");
  state.uploading = true;
  updateControls();
  showError();
  try {
    let uploaded = sampleUploads.get(id);
    if (uploaded) {
      const existing = await fetch(`/api/images/${uploaded.path}/info`, { cache: "no-store", signal: AbortSignal.timeout(15000) });
      if (existing.status === 404) { sampleUploads.delete(id); uploaded = null; }
      else if (!existing.ok) throw new Error("샘플 사진의 서버 연결을 확인해 주세요.");
      else uploaded = { ...uploaded, ...await responseData(existing) };
    }
    if (!uploaded) {
      const source = await fetch(`/static/samples/${id}.jpg`, { signal: AbortSignal.timeout(15000) });
      if (!source.ok) throw new Error("샘플 사진을 불러오지 못했습니다. 다시 선택해 주세요.");
      uploaded = { ...await storeImage(await source.blob(), sample.label), sampleId: id };
      sampleUploads.set(id, uploaded);
    }
    if (nextPreset) applyPreset(nextPreset, uploaded);
    else { attachImage(uploaded); renderImage(); edited(); }
    toast(`${sample.label} 사진을 넣었습니다.`);
  } catch (error) {
    $("preset").value = state.presetKey;
    showError(error.message || "샘플 사진을 준비하지 못했습니다. 다시 선택해 주세요.");
  } finally { state.uploading = false; updateControls(); }
}

async function responseData(response) {
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = data?.detail;
    const message = Array.isArray(detail) ? detail.map((item) => `${item.loc?.join(" → ") || "입력"}: ${item.msg}`).join("\n") : detail;
    throw new Error(message || `요청을 처리하지 못했습니다. HTTP ${response.status}`);
  }
  if (!data) throw new Error("서버 응답을 읽지 못했습니다.");
  return data;
}

async function uploadImage(file) {
  if (!file || state.running || state.uploading) return;
  if (file.size > 20 * 1024 * 1024) { showError("사진은 20MB 이하로 선택하세요."); return; }
  state.uploading = true;
  updateControls();
  showError();
  try {
    attachImage(await storeImage(file, file.name || "붙여넣은 사진"));
    renderImage();
    edited();
  } catch (error) { showError(error.message || "사진을 준비하지 못했습니다."); }
  finally { state.uploading = false; $("image-input").value = ""; updateControls(); }
}

function switchView(view) {
  state.view = view;
  ["result", "request", "response"].forEach((name) => {
    $(`view-${name}`).hidden = name !== view;
    $(`tab-${name}`).classList.toggle("active", name === view);
    $(`tab-${name}`).setAttribute("aria-selected", String(name === view));
    $(`tab-${name}`).tabIndex = name === view ? 0 : -1;
  });
  $("copy-json").hidden = view === "result" || (view === "response" && !state.last);
}

function percent(value) { return `${(value * 100).toFixed(1)}%`; }
function milliseconds(value) { return Number.isFinite(value) ? `${value.toFixed(1)} ms` : "—"; }

function draftSnapshot(image = state.image, sourceImage = state.sourceImage) {
  return structuredClone({ text: state.text, questions: state.questions, image, sourceImage,
    resolution: image?.long_edge || "original", presetKey: state.presetKey });
}

async function requestDecision(request, draft) {
  const started = performance.now();
  const response = await fetch("/v1/systemone", { method: "POST",
    headers: { "Content-Type": "application/json", "X-Veyra-Playground": "1" },
    body: JSON.stringify(request) });
  const result = await responseData(response);
  const elapsed = performance.now() - started;
  const duration = response.headers.get("X-Veyra-Inference-Ms");
  return { request, response: result, draft, at: new Date().toISOString(),
    inferenceMs: duration === null ? null : Number(duration), roundTripMs: elapsed };
}

function restoreRun(run) {
  if (state.running || state.uploading) return;
  state.text = run.draft.text;
  state.presetKey = run.draft.presetKey;
  $("preset").value = state.presetKey;
  state.questions = structuredClone(run.draft.questions);
  attachImage(structuredClone(run.draft.image), structuredClone(run.draft.sourceImage || run.draft.image));
  state.last = run;
  $("state-text").value = state.text;
  renderQuestions();
  renderImage();
  renderResult(run);
  edited();
  switchView("result");
  showError();
}

function renderResult(run) {
  $("empty-state").hidden = true;
  $("result-content").hidden = false;
  const metadata = $("result-meta");
  metadata.replaceChildren();
  const details = [["모델 처리", milliseconds(run.inferenceMs)], ["브라우저 왕복", milliseconds(run.roundTripMs)], ["입력", `${run.response.usage?.input_tokens ?? "—"} tokens`]];
  if (run.draft.image) {
    details.push(["사진", imageDimensions(run.draft.image)]);
    if (run.draft.image.processing) details.push(["모델 입력 예상", imageDimensions(run.draft.image.processing)]);
  }
  details.forEach(([label, value]) => {
    const item = element("span", "", label);
    item.append(element("strong", "", value));
    metadata.append(item);
  });
  const answers = $("answers");
  answers.replaceChildren();
  Object.entries(run.response.answers).forEach(([name, answer]) => {
    const question = run.request.questions[name];
    const card = element("article", "answer-card");
    const heading = element("div", "answer-heading");
    heading.append(element("h3", "answer-name", name), element("span", "type-tag", answer.type), element("span", `decision-status${answer.abstained ? " abstained" : ""}`, answer.abstained ? "보류 · 검토 필요" : "답변"));
    const probabilities = Object.entries(answer.probabilities);
    const top = [...probabilities].sort((a, b) => b[1] - a[1])[0][0];
    const displayValue = answer.type === "choice" ? (question.criteria[answer.choice] || answer.choice) : answer.type === "noul" ? percent(answer.noul) : answer.score.toFixed(2);
    const subtitle = answer.type === "choice" ? `선택 ID · ${answer.choice}` : answer.type === "noul" ? "명제가 참이라고 판단한 확률" : `기대 등급 · 0부터 ${question.criteria.length - 1}까지`;
    card.append(heading, element("p", "decision-label", answer.type === "choice" ? "가장 높은 후보" : answer.type === "noul" ? "참일 확률" : "예측 점수"), element("p", "decision-value", displayValue), element("p", "decision-subtext", subtitle));
    if (answer.type === "choice") probabilities.sort((a, b) => b[1] - a[1]);
    else if (answer.type === "score") probabilities.sort((a, b) => Number(a[0]) - Number(b[0]));
    probabilities.forEach(([key, probability]) => {
      const row = element("div", `probability-row${key === top ? " winner" : ""}`);
      const description = answer.type === "score" ? `${key} · ${question.criteria[Number(key)]}` : answer.type === "noul" ? `${key === "true" ? "참" : "거짓"} · ${question.criteria[key]}` : `${key} · ${question.criteria[key]}`;
      const label = element("div", "probability-label");
      label.append(element("span", "", description), element("strong", "", percent(probability)));
      const track = element("div", "probability-track");
      track.setAttribute("role", "meter");
      track.setAttribute("aria-label", description);
      track.setAttribute("aria-valuemin", "0");
      track.setAttribute("aria-valuemax", "100");
      track.setAttribute("aria-valuenow", String(probability * 100));
      track.setAttribute("aria-valuetext", percent(probability));
      const fill = element("div", "probability-fill");
      fill.style.setProperty("--probability", `${Math.max(0, Math.min(100, probability * 100))}%`);
      track.append(fill);
      row.append(label, track);
      card.append(row);
    });
    const confidence = element("div", "answer-confidence");
    confidence.append(element("span", "", "최대 후보 확률"), element("strong", "", percent(answer.confidence)));
    card.append(confidence);
    answers.append(card);
  });
  $("response-json").textContent = JSON.stringify(run.response, null, 2);
  edited();
  switchView(state.view);
}

function renderHistory() {
  $("history-panel").hidden = !state.history.length;
  const list = $("history-list");
  list.replaceChildren();
  state.history.forEach((run) => {
    const item = button("", "history-item", () => restoreRun(run));
    const clock = element("time", "", new Date(run.at).toLocaleTimeString("ko-KR", { hour: "2-digit", minute: "2-digit", hour12: false }));
    clock.dateTime = run.at;
    const size = run.draft.image ? `${imageDimensions(run.draft.image)} · ` : "";
    item.append(clock, element("span", "", run.request.state.text || "사진 판단"), element("span", "", `${size}${milliseconds(run.inferenceMs)}`));
    list.append(item);
  });
}

async function runDecision(event) {
  event.preventDefault();
  if (state.running || state.uploading || state.phase !== "ready") return;
  let request;
  try { request = makeRequest(true); } catch (error) { showError(error.message); return; }
  if (PRESETS[state.presetKey]?.requiresImage && !state.image) { showError("샘플 사진을 선택하거나 사진 한 장을 추가하세요."); return; }
  const draft = draftSnapshot();
  state.running = true;
  showError();
  updateControls();
  switchView("result");
  try {
    const run = await requestDecision(request, draft);
    state.last = run;
    state.history.unshift(run);
    state.history = state.history.slice(0, 6);
    renderResult(run);
    renderHistory();
  } catch (error) { showError(error.message || "서버 연결을 확인하고 다시 실행하세요."); }
  finally { state.running = false; updateControls(); }
}

function comparisonFingerprint() {
  return JSON.stringify({ source: state.sourceImage?.path, text: state.text, questions: state.questions });
}

function topAnswers(run) {
  return Object.entries(run.response.answers).map(([name, answer]) => {
    const [key, probability] = Object.entries(answer.probabilities).sort((a, b) => b[1] - a[1])[0];
    return { name, key, probability, abstained: answer.abstained };
  });
}

function sameCandidates(run, reference) {
  const signature = (entry) => JSON.stringify(topAnswers(entry).map(({ name, key }) => [name, key]));
  return signature(run) === signature(reference);
}

function median(values) {
  const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!sorted.length) return null;
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function renderComparison() {
  const comparison = state.comparison;
  $("comparison-panel").hidden = !comparison;
  if (!comparison) return;
  $("comparison-status").textContent = comparison.status;
  const body = $("comparison-rows");
  body.replaceChildren();
  const baseline = comparison.rows.find((row) => row.edge === "original")?.run;
  const rows = [...comparison.rows].sort((a, b) => a.image.width * a.image.height - b.image.width * b.image.height);
  rows.forEach((row) => {
    const tr = element("tr");
    const size = element("td");
    size.append(element("strong", "", imageDimensions(row.image)), element("small", "", `${row.edge === "original" ? "원본" : `긴 변 ${row.edge}px`} · ${imageBytes(row.image.bytes)}`));
    const processed = element("td");
    processed.append(element("span", "", imageDimensions(row.image.processing)), element("small", "", `사진 토큰 ${row.image.processing?.visual_tokens ?? "—"}개 / 질문`));
    const decisions = element("td", "comparison-decisions");
    topAnswers(row.run).forEach((answer) => {
      const item = element("div");
      item.append(element("span", "", `${answer.name}: ${answer.key}`), element("strong", "", percent(answer.probability)));
      if (answer.abstained) item.append(element("small", "", "보류 · 검토 필요"));
      decisions.append(item);
    });
    const match = element("td");
    const agrees = baseline && sameCandidates(row.run, baseline);
    match.append(element("span", `comparison-match${agrees ? " agrees" : " differs"}`, row.edge === "original" ? "비교 기준" : baseline ? (agrees ? "같은 후보" : "후보 변경") : "원본 결과 없음"));
    const detail = element("td");
    const open = button("보기", "text-button", () => {
      restoreRun(row.run);
      $("result-content").scrollIntoView({ behavior: "smooth", block: "nearest" });
    });
    open.disabled = state.running || state.uploading;
    open.setAttribute("aria-label", `${imageDimensions(row.image)} 판단 상세 보기`);
    detail.append(open);
    tr.append(size, processed, decisions, match,
      element("td", "comparison-timing", milliseconds(row.inferenceMedianMs)),
      element("td", "comparison-timing", milliseconds(row.roundTripMedianMs)),
      element("td", "comparison-timing", String(row.run.response.usage?.input_tokens ?? "—")), detail);
    body.append(tr);
  });
  edited();
}

async function compareResolutions() {
  if (state.running || state.uploading || state.phase !== "ready" || !state.sourceImage) return;
  let request;
  try { request = makeRequest(true); } catch (error) { showError(error.message); return; }
  const source = structuredClone(state.sourceImage);
  const edges = ["original", ...state.imageLongEdges.filter((edge) => edge < Math.max(source.width, source.height))];
  const baseDraft = draftSnapshot();
  state.comparing = true;
  state.running = true;
  state.compareStop = false;
  state.comparison = { at: new Date().toISOString(), fingerprint: comparisonFingerprint(), source,
    imagePolicy: structuredClone(state.imagePolicy), protocol: { warmups: 1, measuredRuns: 3,
      sequential: true, sourceResize: "Pillow thumbnail LANCZOS; PNG; no crop or upscale",
      excludes: ["image preparation", "upload", "model loading"], accuracyMeasured: false },
    rows: [], complete: false, status: "비교할 사진을 준비하고 있습니다." };
  const comparison = state.comparison;
  showError();
  updateControls();
  renderComparison();
  $("comparison-panel").scrollIntoView({ behavior: "smooth", block: "start" });
  try {
    for (const [index, edge] of edges.entries()) {
      if (state.compareStop) break;
      const image = await resolutionImage(edge, source);
      const label = `${index + 1}/${edges.length} · ${imageDimensions(image)}`;
      const draft = { ...structuredClone(baseDraft), image, sourceImage: source, resolution: edge };
      const variantRequest = { ...request, state: { ...request.state, images: [{ path: image.path }] } };
      if (state.compareStop) break;
      comparison.status = `${label} · 예열 중`;
      renderComparison();
      await requestDecision(variantRequest, draft);
      const samples = [];
      for (let iteration = 0; iteration < 3 && !state.compareStop; iteration += 1) {
        comparison.status = `${label} · 측정 ${iteration + 1}/3`;
        renderComparison();
        samples.push(await requestDecision(variantRequest, draft));
      }
      if (samples.length !== 3) break;
      comparison.rows.push({ edge, image, run: samples[0], samples,
        inferenceMedianMs: median(samples.map((run) => run.inferenceMs)),
        roundTripMedianMs: median(samples.map((run) => run.roundTripMs)) });
      renderComparison();
    }
    comparison.complete = comparison.rows.length === edges.length;
    comparison.status = comparison.complete
      ? `${edges.length}개 크기 비교 완료 · 시간은 3회 중앙값, 판단과 확률은 첫 측정 결과입니다. 이 사진의 비교이며 정확도 평가는 아닙니다.`
      : `비교 중단 · ${comparison.rows.length}/${edges.length}개 크기의 완료된 결과를 남겼습니다.`;
    const current = comparison.rows.find((row) => row.image.path === state.image?.path);
    if (current) { state.last = current.run; renderResult(current.run); }
  } catch (error) {
    comparison.status = `비교가 완료되지 않았습니다. ${comparison.rows.length}/${edges.length}개 크기까지 기록했습니다. ${error.message}`;
    showError(error.message || "해상도 비교를 완료하지 못했습니다.");
  } finally {
    state.comparing = false;
    state.running = false;
    state.compareStop = false;
    updateControls();
    renderComparison();
  }
}

function exportComparison() {
  if (!state.comparison?.rows.length || state.comparing) return;
  const blob = new Blob([JSON.stringify(state.comparison, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = element("a");
  link.href = url;
  link.download = `veyra-resolution-${state.comparison.at.replace(/[:.]/g, "-")}.json`;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function pollStatus() {
  try {
    const response = await fetch("/api/status", { signal: AbortSignal.timeout(8000) });
    const info = await responseData(response);
    state.phase = info.phase;
    const policyChanged = JSON.stringify(state.imagePolicy) !== JSON.stringify(info.image_policy || null)
      || JSON.stringify(state.imageLongEdges) !== JSON.stringify(info.image_long_edges || []);
    state.imagePolicy = info.image_policy || null;
    state.imageLongEdges = info.image_long_edges || [];
    if (policyChanged) renderResolution();
    if (state.phase === "ready" && state.imagePolicy && state.image && !state.image.processing && !state.running && !state.uploading) {
      const path = state.image.path;
      try {
        const metadata = await fetch(`/api/images/${path}/info`, { signal: AbortSignal.timeout(8000) }).then(responseData);
        if (state.image?.path === path) {
          state.image = { ...state.image, ...metadata };
          if (state.sourceImage?.path === path) state.sourceImage = state.image;
          renderImage();
        }
      } catch { /* An expired photo is reported by the next upload or inference action. */ }
    }
    $("connection").className = `connection ${info.phase}`;
    $("connection-label").textContent = info.phase === "ready" ? (info.busy ? "판단 중" : "모델 연결됨") : info.phase === "error" ? "모델 준비 실패" : "모델 준비 중";
    $("device-label").textContent = info.device && info.device !== "cuda" ? `${info.device} · 모델 서버에서 실행` : "Qwen3.5-2B · 메모리에 준비 중";
    if (info.checkpoint !== "veyra-backbone-recovery-v13") $("model-name").textContent = info.checkpoint;
    if (info.phase === "error") showError(`모델을 준비하지 못했습니다.\n${info.error}`, "connection");
    else if (state.errorKind === "connection") showError();
  } catch {
    state.phase = "offline";
    $("connection").className = "connection error";
    $("connection-label").textContent = "서버 연결 끊김";
    showError("서버에 연결할 수 없습니다. 모델 서버 실행 상태와 Tailscale 연결을 확인하세요.", "connection");
  } finally {
    updateControls();
    setTimeout(pollStatus, state.phase === "loading" ? 1500 : 5000);
  }
}

$("state-text").value = state.text;
$("state-text").addEventListener("input", (event) => { state.text = event.target.value; edited(); });
$("preset").addEventListener("change", (event) => selectPreset(event.target.value));
document.querySelectorAll("[data-photo-preset]").forEach((control) => {
  control.addEventListener("click", () => selectPreset(control.dataset.photoPreset));
});
$("question-add").addEventListener("click", () => {
  if (state.questions.length >= 4) return;
  let index = state.questions.length + 1;
  while (state.questions.some((question) => question.name === `question_${index}`)) index += 1;
  state.questions.push({ name: `question_${index}`, type: "choice", instructions: "", options: [{ key: "a", text: "" }, { key: "b", text: "" }] });
  renderQuestions();
  edited();
  $(`instructions-${state.questions.length - 1}`).focus();
});
$("image-add").addEventListener("click", () => $("image-input").click());
$("image-input").addEventListener("change", (event) => uploadImage(event.target.files[0]));
$("image-remove").addEventListener("click", () => { attachImage(null); renderImage(); edited(); updateControls(); });
$("resolution-compare").addEventListener("click", compareResolutions);
$("comparison-cancel").addEventListener("click", () => { state.compareStop = true; updateControls(); });
$("comparison-export").addEventListener("click", exportComparison);
$("upload-zone").addEventListener("dragover", (event) => { event.preventDefault(); $("upload-zone").classList.add("dragging"); });
$("upload-zone").addEventListener("dragleave", () => $("upload-zone").classList.remove("dragging"));
$("upload-zone").addEventListener("drop", (event) => { event.preventDefault(); $("upload-zone").classList.remove("dragging"); uploadImage(event.dataTransfer.files[0]); });
$("decision-form").addEventListener("paste", (event) => {
  const image = [...(event.clipboardData?.files || [])].find((file) => file.type.startsWith("image/"));
  if (image) { event.preventDefault(); uploadImage(image); }
});
$("decision-form").addEventListener("submit", runDecision);
document.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter" && !$("metrics-dialog").open) {
    event.preventDefault();
    if (!$("run-button").disabled) $("decision-form").requestSubmit();
  }
});
document.querySelectorAll("[data-view]").forEach((tab, index, tabs) => {
  tab.addEventListener("click", () => switchView(tab.dataset.view));
  tab.addEventListener("keydown", (event) => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    const next = event.key === "Home" ? 0 : event.key === "End" ? tabs.length - 1 : (index + (event.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
    switchView(tabs[next].dataset.view);
    tabs[next].focus();
  });
});
$("copy-json").addEventListener("click", async () => {
  const text = state.view === "request" ? $("request-json").textContent : $("response-json").textContent;
  try { await navigator.clipboard.writeText(text); toast("JSON을 복사했습니다."); }
  catch { toast("복사 권한을 확인하거나 JSON 텍스트를 직접 선택해 복사하세요."); }
});
$("history-clear").addEventListener("click", () => { state.history = []; renderHistory(); });
$("guide-open").addEventListener("click", () => $("metrics-dialog").showModal());
$("guide-close").addEventListener("click", () => $("metrics-dialog").close());
renderQuestions();
renderSamples();
renderImage();
edited();
pollStatus();
