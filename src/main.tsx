import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import Overlay from "./Overlay";
import MeetingPrompt from "./MeetingPrompt";
import MeetingIndicator from "./MeetingIndicator";
import "@fontsource-variable/inter";
import "@fontsource-variable/source-serif-4";
import "@fontsource-variable/jetbrains-mono";
import "./styles/index.css";

const route = window.location.hash;

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    {route === "#overlay" ? <Overlay /> : route === "#prompt" ? <MeetingPrompt /> : route === "#meeting-indicator" ? <MeetingIndicator /> : <App />}
  </React.StrictMode>,
);
