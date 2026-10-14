# Frontend (static dashboard, hosted on AWS Amplify)

Plain HTML/CSS/JS plus Cytoscape.js; no build step. Signed-out visitors see a landing page with live graph statistics and an in-page
sign-in / sign-up card (Cognito user-pool API, email verification, password reset). Signed-in users can **Ask** (GraphRAG answers
with clickable citations), **Explore** (interactive graph) and, as project members, **upload papers** with live progress.

Setup: copy `config.example.js` to `config.js` (git-ignored) and fill in the stack outputs. Run locally with
`python -m http.server 8080 --directory src/frontend`. To deploy, zip `index.html, app.js, style.css, config.js` and upload with
Amplify's "Deploy without Git".
