(() => {
  const form = document.getElementById("ask-form");
  const question = document.getElementById("question");
  const ticketId = document.getElementById("ticket-id");
  const messages = document.getElementById("messages");
  const welcome = document.getElementById("welcome");
  const typing = document.getElementById("typing");
  const sendButton = document.getElementById("send-button");
  const conversation = document.getElementById("conversation");
  const sessionKey = "ola-support-session-id";

  const newSessionId = () => {
    if (window.crypto && typeof window.crypto.randomUUID === "function") {
      return window.crypto.randomUUID();
    }
    return `web-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  };

  let sessionId = sessionStorage.getItem(sessionKey);
  if (!sessionId) {
    sessionId = newSessionId();
    sessionStorage.setItem(sessionKey, sessionId);
  }

  const addMessage = (role, text, response) => {
    const row = document.createElement("article");
    row.className = `message ${role}`;

    if (role === "assistant") {
      const avatar = document.createElement("span");
      avatar.className = "message-avatar";
      avatar.setAttribute("aria-hidden", "true");
      avatar.textContent = "o";
      row.append(avatar);
    }

    const body = document.createElement("div");
    body.className = "message-body";
    body.append(document.createTextNode(text));

    const meta = document.createElement("small");
    meta.className = "message-meta";
    meta.textContent = role === "user" ? "You" : "Ola Support Assistant";
    body.append(meta);

    if (response && Array.isArray(response.citations) && response.citations.length) {
      const details = document.createElement("details");
      details.className = "sources";
      const summary = document.createElement("summary");
      summary.textContent = `Sources (${response.citations.length})`;
      const list = document.createElement("ul");
      for (const citation of response.citations) {
        const item = document.createElement("li");
        item.textContent = citation.doc_id || "Support policy";
        list.append(item);
      }
      details.append(summary, list);
      body.append(details);
    }

    if (response && response.ticket_info && typeof response.ticket_info === "object") {
      const ticket = document.createElement("div");
      ticket.className = "ticket-result";
      const title = document.createElement("strong");
      title.textContent = `Ticket ${response.ticket_info.record_id || ""}`;
      const details = document.createElement("span");
      details.textContent = `Status: ${response.ticket_info.status || "Unavailable"} · Resolution time: ${response.ticket_info.resolution_time_hours ?? "Unavailable"} hours · Escalation score: ${response.ticket_info.escalation_score ?? "Unavailable"}`;
      ticket.append(title, details);
      body.append(ticket);
    }

    row.append(body);
    messages.append(row);
    conversation.scrollTop = conversation.scrollHeight;
  };

  const submitQuestion = async (text) => {
    const query = text.trim();
    if (!query || sendButton.disabled) return;

    welcome.hidden = true;
    addMessage("user", query);
    question.value = "";
    question.style.height = "auto";
    sendButton.disabled = true;
    typing.hidden = false;
    conversation.scrollTop = conversation.scrollHeight;

    const payload = { query, session_id: sessionId };
    if (ticketId.value.trim()) payload.ticket_id = ticketId.value.trim();

    try {
      const result = await fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await result.json();
      if (!result.ok) {
        throw new Error(data.detail || "The support service could not process that question.");
      }
      if (data.session_id) {
        sessionId = data.session_id;
        sessionStorage.setItem(sessionKey, sessionId);
      }
      addMessage("assistant", data.answer || "I couldn't find an answer just now.", data);
    } catch (error) {
      addMessage("assistant", `Sorry, I couldn't reach support right now. ${error.message}`);
    } finally {
      typing.hidden = true;
      sendButton.disabled = false;
      question.focus();
    }
  };

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    submitQuestion(question.value);
  });

  question.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      form.requestSubmit();
    }
  });

  question.addEventListener("input", () => {
    question.style.height = "auto";
    question.style.height = `${Math.min(question.scrollHeight, 104)}px`;
  });

  document.querySelectorAll("[data-prompt]").forEach((button) => {
    button.addEventListener("click", () => submitQuestion(button.dataset.prompt));
  });

  const clearConversation = () => {
    sessionId = newSessionId();
    sessionStorage.setItem(sessionKey, sessionId);
    messages.replaceChildren();
    welcome.hidden = false;
    ticketId.value = "";
    question.focus();
  };

  document.getElementById("new-chat").addEventListener("click", clearConversation);
  document.getElementById("clear-chat").addEventListener("click", clearConversation);
})();
