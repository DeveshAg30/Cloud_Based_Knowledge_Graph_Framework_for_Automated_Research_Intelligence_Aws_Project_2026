// Research Intelligence dashboard.
// Signed out: landing page with live stats + in-page sign in / sign up (Cognito user-pool API, no redirect).
// Signed in: Ask (GraphRAG), Explore (graph view) and, for project members, Upload.
const cfg = window.APP_CONFIG;
const $ = (id) => document.getElementById(id);
const REGION = cfg.region || (cfg.cognitoDomain.match(/auth\.([a-z0-9-]+)\.amazoncognito/) || [])[1] || "us-east-1";
const COGNITO_URL = `https://cognito-idp.${REGION}.amazonaws.com/`;

const COLORS = {
  Paper: "#4c78a8", Author: "#f58518", Institution: "#b279a2", Venue: "#9d755d",
  Concept: "#72b7b2", Method: "#54a24b", Dataset: "#e45756", Task: "#eeca3b", Metric: "#bab0ac",
};

// ================================================================== storage (wrapped: may be blocked)
function stores() { const s = []; try { s.push(localStorage); } catch (e) {} try { s.push(sessionStorage); } catch (e) {} return s; }
function getItem(k) { for (const s of stores()) { try { const v = s.getItem(k); if (v) return v; } catch (e) {} } return null; }
function setSession(obj, remember) {
  clearSession();
  try { const s = remember ? localStorage : sessionStorage; for (const [k, v] of Object.entries(obj)) s.setItem(k, v); } catch (e) {}
}
function clearSession() {
  for (const s of stores()) for (const k of ["id_token", "access_token", "refresh_token", "expires_at"]) { try { s.removeItem(k); } catch (e) {} }
}

// ================================================================== Cognito user-pool API
async function cognito(action, body) {
  const resp = await fetch(COGNITO_URL, {
    method: "POST",
    headers: { "Content-Type": "application/x-amz-json-1.1", "X-Amz-Target": `AWSCognitoIdentityProviderService.${action}` },
    body: JSON.stringify({ ClientId: cfg.clientId, ...body }),
  });
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    const err = new Error(data.message || data.Message || resp.statusText);
    err.code = (data.__type || "").split("#").pop();
    throw err;
  }
  return data;
}

function friendly(err) {
  const map = {
    NotAuthorizedException: "Incorrect email or password.",
    UserNotConfirmedException: "Please verify your email first. We've sent you a new code.",
    UsernameExistsException: "An account with this email already exists. Try signing in.",
    InvalidPasswordException: "Password doesn't meet the requirements below.",
    CodeMismatchException: "That code isn't right. Check the email and try again.",
    ExpiredCodeException: "That code has expired. Request a new one.",
    LimitExceededException: "Too many attempts. Please wait a few minutes and try again.",
    TooManyRequestsException: "Too many attempts. Please wait a few minutes and try again.",
    InvalidParameterException: "Please check the details you entered.",
    UserNotFoundException: "Incorrect email or password.",
  };
  return map[err.code] || err.message || "Something went wrong. Please try again.";
}

let pending = { email: "", password: "", session: "", remember: true };

function saveTokens(result) {
  setSession({
    id_token: result.IdToken, access_token: result.AccessToken,
    refresh_token: result.RefreshToken || getItem("refresh_token") || "",
    expires_at: String(Date.now() + result.ExpiresIn * 1000),
  }, pending.remember);
}

async function signIn(email, password) {
  const r = await cognito("InitiateAuth", { AuthFlow: "USER_PASSWORD_AUTH", AuthParameters: { USERNAME: email, PASSWORD: password } });
  if (r.ChallengeName === "NEW_PASSWORD_REQUIRED") { pending.session = r.Session; return "newpw"; }
  saveTokens(r.AuthenticationResult);
  return "ok";
}

async function validToken() {
  const t = getItem("id_token");
  if (t && Number(getItem("expires_at")) > Date.now() + 30000) return t;
  const refresh = getItem("refresh_token");
  if (!refresh) return null;
  try {
    const r = await cognito("InitiateAuth", { AuthFlow: "REFRESH_TOKEN_AUTH", AuthParameters: { REFRESH_TOKEN: refresh } });
    pending.remember = !!(function () { try { return localStorage.getItem("refresh_token"); } catch (e) { return null; } })();
    saveTokens(r.AuthenticationResult);
    return r.AuthenticationResult.IdToken;
  } catch (e) { clearSession(); return null; }
}

function claims(t) {
  const b64 = t.split(".")[1].replace(/-/g, "+").replace(/_/g, "/");
  return JSON.parse(decodeURIComponent(escape(atob(b64))));
}

// ================================================================== auth card UI
const VIEWS = ["signin", "signup", "confirm", "forgot", "reset", "newpw"];
function show(view, message, kind = "info") {
  for (const v of VIEWS) $(`view-${v}`).hidden = v !== view;
  document.querySelectorAll(".tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === view));
  document.querySelector(".tabs").hidden = !["signin", "signup"].includes(view);
  document.querySelectorAll(".email-echo").forEach((e) => (e.textContent = pending.email));
  setMsg(message, kind);
  const first = $(`view-${view}`).querySelector("input:not([type=hidden])");
  if (first && window.matchMedia("(min-width: 900px)").matches) first.focus();
}
function setMsg(text, kind = "info") {
  const m = $("auth-msg");
  m.hidden = !text; m.textContent = text || ""; m.className = `msg ${kind}`;
}
function fail(form, err) {
  setMsg(friendly(err), "error");
  form.closest(".auth-card").classList.remove("shake"); void form.offsetWidth;
  form.closest(".auth-card").classList.add("shake");
}
async function busy(form, fn) {
  const btn = form.querySelector("button[type=submit]");
  btn.disabled = true; btn.classList.add("loading");
  try { await fn(); } catch (err) { fail(form, err); } finally { btn.disabled = false; btn.classList.remove("loading"); }
}

const RULES = { len: (p) => p.length >= 10, upper: (p) => /[A-Z]/.test(p), lower: (p) => /[a-z]/.test(p), num: (p) => /\d/.test(p) };
function wirePasswordRules(form) {
  const input = form.querySelector("input[name=password]");
  const rules = form.querySelector(".pw-rules");
  const meter = form.querySelector(".meter span");
  if (!rules) return;
  input.addEventListener("input", () => {
    const p = input.value;
    let passed = 0;
    rules.querySelectorAll("li").forEach((li) => { const ok = RULES[li.dataset.rule](p); li.classList.toggle("ok", ok); passed += ok; });
    if (meter) {
      const score = passed + (p.length >= 14) + /[^A-Za-z0-9]/.test(p);
      meter.style.width = `${Math.min(100, (score / 6) * 100)}%`;
      meter.style.background = score >= 5 ? "var(--ok)" : score >= 3 ? "#e0a800" : "var(--err)";
    }
  });
}
const passwordOk = (p) => Object.values(RULES).every((r) => r(p));

function buildCodeBoxes(form) {
  const wrap = form.querySelector(".code-boxes");
  const hidden = form.querySelector("input[name=code]");
  wrap.innerHTML = "";
  const boxes = Array.from({ length: 6 }, (_, i) => {
    const b = document.createElement("input");
    b.inputMode = "numeric"; b.maxLength = 1; b.autocomplete = i === 0 ? "one-time-code" : "off"; b.ariaLabel = `Digit ${i + 1}`;
    wrap.appendChild(b);
    return b;
  });
  const sync = () => { hidden.value = boxes.map((b) => b.value).join(""); if (hidden.value.length === 6) form.requestSubmit(); };
  boxes.forEach((b, i) => {
    b.addEventListener("input", () => {
      const digits = b.value.replace(/\D/g, "");
      if (digits.length > 1) {  // pasted or autofilled the whole code
        digits.slice(0, 6).split("").forEach((d, j) => { if (boxes[j]) boxes[j].value = d; });
        boxes[Math.min(5, digits.length - 1)].focus();
      } else { b.value = digits; if (digits && boxes[i + 1]) boxes[i + 1].focus(); }
      sync();
    });
    b.addEventListener("keydown", (e) => { if (e.key === "Backspace" && !b.value && boxes[i - 1]) boxes[i - 1].focus(); });
  });
  boxes[0].focus();
}

let resendTimer = null;
function startResendCooldown() {
  const btn = $("resend-btn");
  let left = 30;
  btn.disabled = true;
  clearInterval(resendTimer);
  resendTimer = setInterval(() => {
    btn.textContent = left > 0 ? `Resend code (${left--}s)` : "Resend code";
    if (left < 0) { clearInterval(resendTimer); btn.disabled = false; }
  }, 1000);
}

function wireAuth() {
  document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => show(b.dataset.tab)));
  document.querySelectorAll("[data-goto]").forEach((a) => a.addEventListener("click", (e) => { e.preventDefault(); show(a.dataset.goto); }));
  document.querySelectorAll(".eye").forEach((btn) => btn.addEventListener("click", () => {
    const input = btn.previousElementSibling;
    input.type = input.type === "password" ? "text" : "password";
    btn.textContent = input.type === "password" ? "Show" : "Hide";
  }));
  ["signup", "reset", "newpw"].forEach((v) => wirePasswordRules($(`view-${v}`)));

  $("view-signin").addEventListener("submit", (e) => {
    e.preventDefault();
    const f = e.target;
    pending = { ...pending, email: f.email.value.trim().toLowerCase(), password: f.password.value, remember: f.remember.checked };
    busy(f, async () => {
      try {
        const r = await signIn(pending.email, pending.password);
        if (r === "newpw") return show("newpw");
        enterApp();
      } catch (err) {
        if (err.code === "UserNotConfirmedException") {
          await cognito("ResendConfirmationCode", { Username: pending.email }).catch(() => {});
          show("confirm", friendly(err)); buildCodeBoxes($("view-confirm")); startResendCooldown();
          return;
        }
        throw err;
      }
    });
  });

  $("view-signup").addEventListener("submit", (e) => {
    e.preventDefault();
    const f = e.target;
    if (!passwordOk(f.password.value)) return fail(f, { code: "InvalidPasswordException" });
    pending = { ...pending, email: f.email.value.trim().toLowerCase(), password: f.password.value, remember: true };
    const attrs = [{ Name: "email", Value: pending.email }];
    if (f.name.value.trim()) attrs.push({ Name: "name", Value: f.name.value.trim() });
    busy(f, async () => {
      await cognito("SignUp", { Username: pending.email, Password: pending.password, UserAttributes: attrs });
      show("confirm", "Account created. Enter the code we emailed you."); buildCodeBoxes($("view-confirm")); startResendCooldown();
    });
  });

  $("view-confirm").addEventListener("submit", (e) => {
    e.preventDefault();
    const f = e.target;
    if (f.code.value.length !== 6) return;
    busy(f, async () => {
      await cognito("ConfirmSignUp", { Username: pending.email, ConfirmationCode: f.code.value });
      if (pending.password) { await signIn(pending.email, pending.password); enterApp(); }
      else show("signin", "Email verified. Please sign in.");
    });
  });
  $("resend-btn").addEventListener("click", async () => {
    try { await cognito("ResendConfirmationCode", { Username: pending.email }); setMsg("A new code is on its way."); startResendCooldown(); }
    catch (err) { setMsg(friendly(err), "error"); }
  });

  $("view-forgot").addEventListener("submit", (e) => {
    e.preventDefault();
    const f = e.target;
    pending.email = f.email.value.trim().toLowerCase();
    busy(f, async () => {
      await cognito("ForgotPassword", { Username: pending.email });
      show("reset", "If an account exists for this email, a reset code is on its way."); buildCodeBoxes($("view-reset"));
    });
  });

  $("view-reset").addEventListener("submit", (e) => {
    e.preventDefault();
    const f = e.target;
    if (f.code.value.length !== 6) return fail(f, { message: "Enter the 6-digit code from the email." });
    if (!passwordOk(f.password.value)) return fail(f, { code: "InvalidPasswordException" });
    busy(f, async () => {
      await cognito("ConfirmForgotPassword", { Username: pending.email, ConfirmationCode: f.code.value, Password: f.password.value });
      show("signin", "Password updated. Sign in with your new password.");
      $("view-signin").email.value = pending.email;
    });
  });

  $("view-newpw").addEventListener("submit", (e) => {
    e.preventDefault();
    const f = e.target;
    if (!passwordOk(f.password.value)) return fail(f, { code: "InvalidPasswordException" });
    busy(f, async () => {
      const r = await cognito("RespondToAuthChallenge", {
        ChallengeName: "NEW_PASSWORD_REQUIRED", Session: pending.session,
        ChallengeResponses: { USERNAME: pending.email, NEW_PASSWORD: f.password.value },
      });
      saveTokens(r.AuthenticationResult);
      enterApp();
    });
  });

  // Example questions: signed-out visitors are nudged to sign up, and the question runs right after login.
  document.querySelectorAll(".chip").forEach((chip) => chip.addEventListener("click", () => {
    try { sessionStorage.setItem("pending_question", chip.textContent); } catch (e) {}
    show("signup", "Create a free account (or sign in) and we'll run that question for you.");
    $("auth-card").scrollIntoView({ behavior: "smooth", block: "center" });
  }));
}

// ================================================================== landing: live stats + animated graph
async function loadPublicStats() {
  try {
    const resp = await fetch(`${cfg.apiUrl}/public/stats`);
    if (!resp.ok) return;
    const data = await resp.json();
    document.querySelectorAll("[data-count]").forEach((el) => countUp(el, data[el.dataset.count] || 0));
  } catch (e) { /* landing still works without numbers */ }
}
function countUp(el, target) {
  const start = performance.now(), dur = 1200;
  const step = (now) => {
    const t = Math.min(1, (now - start) / dur);
    el.textContent = Math.round(target * (1 - Math.pow(1 - t, 3))).toLocaleString();
    if (t < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

let bgAnim = null;
function startBackground() {
  const canvas = $("bg-graph");
  const ctx = canvas.getContext("2d");
  const palette = Object.values(COLORS);
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  let nodes = [], w = 0, h = 0;
  const resize = () => {
    const dpr = window.devicePixelRatio || 1;
    w = canvas.clientWidth; h = canvas.clientHeight;
    canvas.width = w * dpr; canvas.height = h * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const n = Math.round(Math.min(70, (w * h) / 18000));
    nodes = Array.from({ length: n }, () => ({
      x: Math.random() * w, y: Math.random() * h, vx: (Math.random() - .5) * .35, vy: (Math.random() - .5) * .35,
      r: 2 + Math.random() * 3, c: palette[Math.floor(Math.random() * palette.length)],
    }));
  };
  const mouse = { x: -999, y: -999 };
  window.addEventListener("mousemove", (e) => { const r = canvas.getBoundingClientRect(); mouse.x = e.clientX - r.left; mouse.y = e.clientY - r.top; });
  const frame = () => {
    ctx.clearRect(0, 0, w, h);
    for (const a of nodes) {
      if (!reduce) { a.x += a.vx; a.y += a.vy; if (a.x < 0 || a.x > w) a.vx *= -1; if (a.y < 0 || a.y > h) a.vy *= -1; }
    }
    for (let i = 0; i < nodes.length; i++) for (let j = i + 1; j < nodes.length; j++) {
      const a = nodes[i], b = nodes[j], d = Math.hypot(a.x - b.x, a.y - b.y);
      if (d < 130) { ctx.strokeStyle = `rgba(130,145,170,${(1 - d / 130) * .35})`; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke(); }
    }
    for (const a of nodes) {
      const near = Math.hypot(a.x - mouse.x, a.y - mouse.y) < 90;  // nodes light up near the cursor
      ctx.fillStyle = a.c; ctx.globalAlpha = near ? 1 : .75;
      ctx.beginPath(); ctx.arc(a.x, a.y, near ? a.r + 2 : a.r, 0, Math.PI * 2); ctx.fill();
    }
    ctx.globalAlpha = 1;
    if (!reduce) bgAnim = requestAnimationFrame(frame);
  };
  resize(); window.addEventListener("resize", resize); frame();
}
function stopBackground() { if (bgAnim) cancelAnimationFrame(bgAnim); bgAnim = null; }

// ================================================================== app (signed in)
async function api(path, options = {}) {
  const t = await validToken();
  if (!t) { showLanding("Your session expired. Please sign in again."); throw new Error("session expired"); }
  const resp = await fetch(cfg.apiUrl + path, {
    ...options,
    headers: { Authorization: `Bearer ${t}`, "Content-Type": "application/json", ...(options.headers || {}) },
  });
  if (resp.status === 401) { clearSession(); showLanding("Your session expired. Please sign in again."); throw new Error("session expired"); }
  if (resp.status === 429) throw new Error("The service is busy. Please wait a few seconds and try again.");
  const body = await resp.json().catch(() => ({}));
  if (!resp.ok) throw new Error(body.error || resp.statusText);
  return body;
}

let cy = null;
function initGraph() {
  if (cy) return;
  cy = cytoscape({
    container: $("cy"),
    style: [
      { selector: "node", style: {
        "background-color": (n) => COLORS[n.data("label")] || "#888", label: "data(short)",
        "font-size": 9, color: getComputedStyle(document.body).color, "text-wrap": "ellipsis", "text-max-width": 110,
        width: 18, height: 18 } },
      { selector: "node[?seed]", style: { width: 30, height: 30, "border-width": 3, "border-color": "#222" } },
      { selector: "edge", style: {
        width: 1.2, "line-color": "#9aa3b2", "target-arrow-color": "#9aa3b2", "target-arrow-shape": "triangle",
        "curve-style": "bezier", label: "data(rel)", "font-size": 7, color: "#9aa3b2", "text-rotation": "autorotate" } },
    ],
  });
  cy.on("tap", "node", (evt) => { $("entity").value = evt.target.data("name"); explore(); });
  $("legend").innerHTML = Object.entries(COLORS).map(([k, c]) => `<span style="--c:${c}">${k}</span>`).join("");
}

function drawGraph(nodes, edges) {
  cy.elements().remove();
  const ids = new Set(nodes.map((n) => n.id));
  cy.add(nodes.map((n) => ({ data: { ...n, short: n.name.length > 40 ? n.name.slice(0, 40) + "…" : n.name } })));
  cy.add(edges.filter((e) => ids.has(e.source) && ids.has(e.target)).map((e, i) => ({ data: { id: `e${i}`, ...e } })));
  cy.layout({ name: "cose", animate: false, nodeRepulsion: 9000, idealEdgeLength: 90 }).run();
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// Minimal markdown for LLM answers (bold, italics, bullet lists) plus clickable citations.
// Input is escaped first, so only these few tags can ever appear.
function citeLink(id) {
  const safe = id.trim();
  return /^W\d+$/.test(safe)
    ? `<a class="cite" href="https://openalex.org/${safe}" target="_blank" rel="noopener">${safe}</a>`
    : `<span class="cite">${safe}</span>`;
}
function renderAnswer(text) {
  const lines = escapeHtml(text).split("\n");
  let html = "", inList = false;
  for (const raw of lines) {
    const bullet = raw.match(/^\s*[*-]\s+(.*)$/);
    if (bullet && !inList) { html += "<ul>"; inList = true; }
    if (!bullet && inList) { html += "</ul>"; inList = false; }
    const line = (bullet ? bullet[1] : raw)
      .replace(/\*\*(.+?)\*\*/g, "<b>$1</b>")
      .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<i>$2</i>")
      .replace(/\[(?:doc:)?((?:W\d+|[\w.-]+-\d+|s2-\d+)(?:\s*,\s*(?:doc:)?(?:W\d+|[\w.-]+-\d+|s2-\d+))*)\]/g,
        (_, ids) => "[" + ids.split(",").map((i) => citeLink(i.replace("doc:", ""))).join(", ") + "]");
    html += bullet ? `<li>${line}</li>` : (line.trim() ? `<p>${line}</p>` : "");
  }
  return html + (inList ? "</ul>" : "");
}

// Bumped on sign-out so a request that finishes afterwards can't paint into the next user's screen.
let sessionGen = 0;

function resetApp() {
  sessionGen++;
  for (const id of ["question", "entity"]) $(id).value = "";
  for (const id of ["answer", "facts", "stats", "upload-log"]) $(id).innerHTML = "";
  $("user-name").textContent = "";
  $("member-badge").hidden = true;
  $("upload-member").hidden = true;
  $("upload-public").hidden = false;
  document.querySelector("#ask-panel details").open = false;
  if (cy) cy.elements().remove();
  document.querySelectorAll("form.view").forEach((f) => f.reset());
  document.querySelectorAll(".pw-rules li").forEach((li) => li.classList.remove("ok"));
  document.querySelectorAll(".meter span").forEach((m) => (m.style.width = "0"));
  pending = { email: "", password: "", session: "", remember: true };
  try { sessionStorage.removeItem("pending_question"); } catch (e) {}
}

async function ask(evt) {
  if (evt) evt.preventDefault();
  const btn = $("ask-form").querySelector("button"); btn.disabled = true; btn.classList.add("loading");
  $("answer").textContent = "Searching the graph and writing an answer…"; $("facts").innerHTML = "";
  const gen = sessionGen;
  try {
    const r = await api("/query", { method: "POST", body: JSON.stringify({ question: $("question").value }) });
    if (gen !== sessionGen) return;
    $("answer").innerHTML = renderAnswer(r.answer);
    $("facts").innerHTML = r.facts.map((f) => `<li>${escapeHtml(f.text)}${f.doc_id ? ` [${citeLink(escapeHtml(f.doc_id))}]` : ""}</li>`).join("");
    drawGraph(r.graph.nodes, r.graph.edges);
  } catch (e) {
    if (gen === sessionGen) $("answer").textContent = `Error: ${e.message}`;
  } finally { btn.disabled = false; btn.classList.remove("loading"); }
}

async function explore(evt) {
  if (evt) evt.preventDefault();
  const gen = sessionGen;
  try {
    const r = await api(`/graph?name=${encodeURIComponent($("entity").value)}`);
    if (gen === sessionGen) drawGraph(r.nodes, r.edges);
  } catch (e) { if (gen === sessionGen) alert(e.message); }
}

async function loadStats() {
  const gen = sessionGen;
  const r = await api("/stats");
  if (gen !== sessionGen) return;
  $("stats").innerHTML = Object.entries(r.counts).sort((a, b) => b[1] - a[1])
    .map(([k, v]) => `<li><span>${escapeHtml(k)}</span><b>${Number(v).toLocaleString()}</b></li>`).join("") || "<li>empty graph</li>";
}

// Upload card: S3 upload -> ingest Lambda (+ SciBERT on SageMaker) -> knowledge graph, shown live.
const STEPS = ["Upload to Amazon S3", "Extract entities in the cloud", "Add to the knowledge graph"];

function uploadCard(name) {
  const li = document.createElement("li");
  li.className = "up-card";
  li.innerHTML = `<div class="up-name">${escapeHtml(name)}</div>
    <ol class="up-steps">${STEPS.map((s) => `<li>${s}</li>`).join("")}</ol>
    <div class="up-timer"></div><div class="up-result"></div>`;
  $("upload-log").prepend(li);
  return li;
}
function setStep(li, n, state = "active") {
  li.querySelectorAll(".up-steps li").forEach((s, i) => {
    s.className = i < n ? "done" : i === n ? state : "";
  });
}

async function upload(evt) {
  for (const file of evt.target.files) {
    const li = uploadCard(file.name);
    setStep(li, 0);
    try {
      const { url, key } = await api("/upload-url", { method: "POST", body: JSON.stringify({ filename: file.name }) });
      const put = await fetch(url, { method: "PUT", body: file });
      if (!put.ok) throw new Error("upload to S3 failed");
      setStep(li, 1);
      trackUpload(li, key);
    } catch (e) { setStep(li, 0, "failed"); li.querySelector(".up-result").textContent = e.message; }
  }
  evt.target.value = "";
}

async function trackUpload(li, key) {
  const gen = sessionGen, started = Date.now();
  const timer = li.querySelector(".up-timer");
  const tick = setInterval(() => {
    const s = Math.round((Date.now() - started) / 1000);
    timer.textContent = `${s}s${s > 45 ? " (first upload after a quiet period loads the model, up to a minute)" : ""}`;
  }, 1000);
  try {
    for (let i = 0; i < 60 && gen === sessionGen; i++) {   // up to ~5 minutes
      await new Promise((r) => setTimeout(r, 5000));
      if (gen !== sessionGen) return;
      const st = await api(`/upload-status?key=${encodeURIComponent(key)}`);
      if (st.status === "processing") continue;
      if (st.status === "failed") throw new Error(st.error || "processing failed");
      setStep(li, 3);
      timer.textContent = `Done in ${st.seconds}s · ${st.edges} graph edges added`;
      renderUploadResult(li, st);
      loadStats().catch(() => {});
      return;
    }
    if (gen === sessionGen) throw new Error("Still processing. Check again in a minute.");
  } catch (e) {
    setStep(li, 1, "failed"); li.querySelector(".up-result").textContent = e.message;
  } finally { clearInterval(tick); }
}

function renderUploadResult(li, st) {
  const byType = {};
  for (const e of st.entities) (byType[e.label] = byType[e.label] || []).push(e.text);
  const groups = Object.entries(byType).map(([label, names]) =>
    `<div class="ent-group"><span class="ent-label" style="--c:${COLORS[label] || "#888"}">${escapeHtml(label)} (${names.length})</span>
     ${names.slice(0, 12).map((n) => `<span class="ent" style="--c:${COLORS[label] || "#888"}">${escapeHtml(n)}</span>`).join("")}</div>`).join("");
  li.querySelector(".up-result").innerHTML = `
    <div class="up-title">${escapeHtml(st.title || "Untitled")}</div>
    ${st.authors && st.authors.length ? `<div class="up-authors">${escapeHtml(st.authors.join(", "))}</div>` : ""}
    <div class="up-by">Extracted by <b>${escapeHtml(st.extractor)}</b>: ${st.entities.length} entities, ${st.relations.length} relations</div>
    ${groups || '<div class="up-by">No methods, datasets, tasks or metrics found.</div>'}
    <button class="primary small">Explore in graph</button>`;
  li.querySelector("button").onclick = () => {
    $("entity").value = st.title || "";
    explore();
    $("graph-panel").scrollIntoView({ behavior: "smooth" });
  };
}

let appWired = false;
async function enterApp() {
  const t = await validToken();
  if (!t) return showLanding();
  const c = claims(t);
  const member = String(c["cognito:groups"] || "").includes("members");
  $("user-name").textContent = c.name || c.email || "";
  $("member-badge").hidden = !member;
  $("upload-member").hidden = !member;
  $("upload-public").hidden = member;
  $("landing").hidden = true; $("app").hidden = false; $("user-box").hidden = false;
  stopBackground();
  initGraph(); cy.resize();
  if (!appWired) {
    $("ask-form").onsubmit = ask;
    $("explore-form").onsubmit = explore;
    $("upload").onchange = upload;
    appWired = true;
  }
  loadStats().catch((e) => { $("stats").innerHTML = `<li>${escapeHtml(e.message)}</li>`; });
  let q = null;
  try { q = sessionStorage.getItem("pending_question"); sessionStorage.removeItem("pending_question"); } catch (e) {}
  if (q) { $("question").value = q; ask(); }
}

function showLanding(message) {
  resetApp();
  $("app").hidden = true; $("user-box").hidden = true; $("landing").hidden = false;
  show("signin", message || "", message ? "error" : "info");
  if (!bgAnim) startBackground();
}

async function signOut() {
  const access = getItem("access_token");
  clearSession();
  if (access) cognito("GlobalSignOut", { AccessToken: access }).catch(() => {});
  showLanding();
  setMsg("You've been signed out.");
}

// ================================================================== boot
(async () => {
  wireAuth();
  $("logout-btn").onclick = signOut;
  if (await validToken()) enterApp();
  else { showLanding(); loadPublicStats(); }
})();
