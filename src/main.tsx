import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import Overlay from "./Overlay";
import MeetingPrompt from "./MeetingPrompt";

const route = window.location.hash;

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    {route === "#overlay" ? <Overlay /> : route === "#prompt" ? <MeetingPrompt /> : <App />}
  </React.StrictMode>,
);
