import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import KeyGate from "./KeyGate";
import "./index.css";
import "./uikit.css";

// KeyGate wraps App rather than living inside it: every screen App can show
// makes API calls to decide what to render, so the key has to be established
// before App mounts at all. On a deployment with no keys configured the gate
// resolves to a pass-through and nothing is ever prompted.
ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <KeyGate>
      <App />
    </KeyGate>
  </React.StrictMode>
);
