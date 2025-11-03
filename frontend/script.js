const API = (path) => `http://127.0.0.1:8000${path}`;

const chat = document.getElementById("chat");
const form = document.getElementById("composer");
const input = document.getElementById("input");

function addMsg(role, text) {
  const wrap = document.createElement("div");
  wrap.className = `msg ${role}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  wrap.appendChild(bubble);
  chat.appendChild(wrap);
  chat.scrollTop = chat.scrollHeight;
  return bubble;
}

function addTyping() {
  const wrap = document.createElement("div");
  wrap.className = "msg bot";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = `<span class="typing"><span class="dot"></span><span class="dot"></span><span class="dot"></span></span>`;
  wrap.appendChild(bubble);
  chat.appendChild(wrap);
  chat.scrollTop = chat.scrollHeight;
  return bubble;
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = input.value.trim();
  if (!q) return;

  addMsg("user", q);
  input.value = "";
  const submitBtn = form.querySelector("button");
  submitBtn.disabled = true;
  const bubble = addTyping();

  try {
    const res = await fetch(API("/chat"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: q })
    });

    if (!res.ok) {
      const errorText = await res.text();
      bubble.textContent = `Server error: ${res.status} - ${errorText}`;
      return;
    }

    if (!res.body) {
      bubble.textContent = "No response body from server.";
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    bubble.textContent = "";
    
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      bubble.textContent += decoder.decode(value, { stream: true });
      chat.scrollTop = chat.scrollHeight;
    }
    
  } catch (err) {
    console.error("Fetch error:", err);
    bubble.textContent = `Error: ${err.message}. Is the server running?`;
  } finally {
    submitBtn.disabled = false;
  }
});

// Test server connection on load
fetch(API("/health"))
  .then(r => r.json())
  .then(data => console.log("✓ Server health check:", data))
  .catch(err => console.error("✗ Server not reachable:", err));