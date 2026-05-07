(function () {
  const { useEffect, useMemo, useRef, useState } = React;
  const h = React.createElement;

  const STORAGE_KEY = "mylabs-studio-chats-v2";
  const MODES = ["RAG Only", "RAG + Model", "Model Only", "SQL RDBMS"];
  const DEFAULT_MODE = "RAG + Model";

  function newChat() {
    const id = crypto.randomUUID ? crypto.randomUUID() : `chat-${Date.now()}-${Math.random()}`;
    return { id, name: "New Chat", messages: [] };
  }

  function chatName(message) {
    const words = message.trim().split(/\s+/).slice(0, 6);
    return words.join(" ") + (message.trim().split(/\s+/).length > 6 ? "..." : "");
  }

  function textValue(value) {
    if (value == null) return "";
    if (typeof value === "string") return value;
    if (typeof value === "number" || typeof value === "boolean") return String(value);
    if (typeof value === "object" && typeof value.content === "string") return value.content;
    try {
      return JSON.stringify(value, null, 2);
    } catch {
      return String(value);
    }
  }

  async function api(path, options) {
    const res = await fetch(path, {
      headers: { "Content-Type": "application/json", ...(options && options.headers) },
      ...options,
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(body.detail || res.statusText);
    }
    return res.json();
  }

  async function readEventStream(res, handlers) {
    if (!res.ok) {
      const body = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(body.detail || res.statusText);
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const frames = buffer.split("\n\n");
      buffer = frames.pop() || "";

      for (const frame of frames) {
        const lines = frame.split("\n");
        const event = (lines.find((line) => line.startsWith("event: ")) || "event: message").slice(7);
        const dataLine = lines.find((line) => line.startsWith("data: "));
        if (!dataLine) continue;
        const data = JSON.parse(dataLine.slice(6));
        if (handlers[event]) handlers[event](data);
      }
    }
  }

  function readSseFrames(buffer, handlers) {
    const frames = buffer.split("\n\n");
    const remainder = frames.pop() || "";

    for (const frame of frames) {
      const lines = frame.split("\n");
      const event = (lines.find((line) => line.startsWith("event: ")) || "event: message").slice(7);
      const dataLine = lines.find((line) => line.startsWith("data: "));
      if (!dataLine) continue;
      const data = JSON.parse(dataLine.slice(6));
      if (handlers[event]) handlers[event](data);
    }

    return remainder;
  }

  function formatBytes(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  function App() {
    const [boot, setBoot] = useState(null);
    const [datasets, setDatasets] = useState([]);
    const [datasetDetail, setDatasetDetail] = useState(null);
    const [activeDataset, setActiveDataset] = useState("");
    const [mode, setMode] = useState(DEFAULT_MODE);
    const [chats, setChats] = useState(() => {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (!saved) return [newChat()];
      try {
        const parsed = JSON.parse(saved);
        return parsed.length ? parsed : [newChat()];
      } catch {
        return [newChat()];
      }
    });
    const [currentChatId, setCurrentChatId] = useState(() => chats[0].id);
    const [input, setInput] = useState("");
    const [stats, setStats] = useState([]);
    const [log, setLog] = useState([]);
    const [busy, setBusy] = useState(false);
    const [newDatasetName, setNewDatasetName] = useState("");
    const [discoverTopic, setDiscoverTopic] = useState("");
    const [discoverMax, setDiscoverMax] = useState(15);
    const [discoverQueries, setDiscoverQueries] = useState(8);
    const chatEndRef = useRef(null);

    const currentChat = useMemo(
      () => chats.find((chat) => chat.id === currentChatId) || chats[0],
      [chats, currentChatId],
    );

    useEffect(() => {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(chats));
    }, [chats]);

    useEffect(() => {
      chatEndRef.current && chatEndRef.current.scrollIntoView({ behavior: "smooth", block: "end" });
    }, [currentChat && currentChat.messages, busy]);

    useEffect(() => {
      api("/api/bootstrap")
        .then((data) => {
          setBoot(data);
          setDatasets(data.datasets || []);
          setActiveDataset(data.active_dataset || "");
          setDatasetDetail(data.dataset_detail || null);
        })
        .catch((err) => setLog((items) => [`Startup error: ${err.message}`, ...items]));
    }, []);

    function patchCurrentChat(fn) {
      setChats((items) => items.map((chat) => (chat.id === currentChat.id ? fn(chat) : chat)));
    }

    function addLog(message) {
      setLog((items) => [message, ...items].slice(0, 120));
    }

    async function selectDataset(name) {
      const data = await api("/api/datasets/select", {
        method: "POST",
        body: JSON.stringify({ name }),
      });
      setActiveDataset(data.active_dataset || "");
      setDatasetDetail(data.dataset_detail || null);
    }

    async function createDataset() {
      const name = newDatasetName.trim();
      if (!name) return;
      const data = await api("/api/datasets", {
        method: "POST",
        body: JSON.stringify({ name }),
      });
      setDatasets(data.datasets || []);
      setDatasetDetail(data.dataset_detail || null);
      setActiveDataset(data.dataset_detail ? data.dataset_detail.name : "");
      setNewDatasetName("");
      addLog(data.message);
    }

    async function deleteDataset() {
      if (!activeDataset || !confirm(`Delete dataset "${activeDataset}"?`)) return;
      const data = await api(`/api/datasets/${encodeURIComponent(activeDataset)}`, { method: "DELETE" });
      setDatasets(data.datasets || []);
      setDatasetDetail(data.dataset_detail || null);
      setActiveDataset(data.dataset_detail ? data.dataset_detail.name : "");
      addLog(data.message);
    }

    async function runDatasetStream(path, options) {
      setBusy(true);
      const firstLog = path === "/api/discover"
        ? "Starting arXiv discovery..."
        : path.includes("/rdbms/generate")
          ? "Generating normalized RDBMS..."
          : "Starting request...";
      setLog([firstLog]);
      const handlers = {
        log: (data) => addLog(data.message),
        error: (data) => addLog(`Error: ${data.message}`),
        done: (data) => {
          setDatasets(data.datasets || []);
          setDatasetDetail(data.dataset_detail || null);
          if (data.dataset_detail) setActiveDataset(data.dataset_detail.name);
          addLog("Complete.");
        },
      };
      try {
        const res = await fetch(path, options);
        await readEventStream(res, handlers);
      } catch (err) {
        addLog(`Request failed: ${err.message}`);
      } finally {
        setBusy(false);
      }
    }

    function uploadDatasetStream(path, form, totalBytes) {
      setBusy(true);
      setLog([`Uploading ${formatBytes(totalBytes)}...`]);

      const handlers = {
        log: (data) => addLog(data.message),
        error: (data) => addLog(`Error: ${data.message}`),
        done: (data) => {
          setDatasets(data.datasets || []);
          setDatasetDetail(data.dataset_detail || null);
          if (data.dataset_detail) setActiveDataset(data.dataset_detail.name);
          addLog("Complete.");
        },
      };

      return new Promise((resolve) => {
        const xhr = new XMLHttpRequest();
        let seen = 0;
        let buffer = "";

        xhr.upload.onprogress = (event) => {
          if (!event.lengthComputable) return;
          const pct = Math.round((event.loaded / event.total) * 100);
          setLog([`Uploading ${formatBytes(event.loaded)} of ${formatBytes(event.total)} (${pct}%)...`]);
        };
        xhr.onprogress = () => {
          buffer = readSseFrames(buffer + xhr.responseText.slice(seen), handlers);
          seen = xhr.responseText.length;
        };
        xhr.onload = () => {
          buffer = readSseFrames(buffer + xhr.responseText.slice(seen), handlers);
          if (xhr.status < 200 || xhr.status >= 300) {
            addLog(`Request failed: ${xhr.statusText || `HTTP ${xhr.status}`}`);
          }
          setBusy(false);
          resolve();
        };
        xhr.onerror = () => {
          addLog("Request failed: network error");
          setBusy(false);
          resolve();
        };
        xhr.open("POST", path);
        xhr.send(form);
      });
    }

    async function uploadFiles(event) {
      const files = Array.from(event.target.files || []);
      event.target.value = "";
      if (!activeDataset || files.length === 0) return;
      const form = new FormData();
      files.forEach((file) => form.append("files", file));
      const totalBytes = files.reduce((total, file) => total + file.size, 0);
      await uploadDatasetStream(`/api/datasets/${encodeURIComponent(activeDataset)}/upload`, form, totalBytes);
    }

    async function reindex() {
      if (!activeDataset) return;
      await runDatasetStream(`/api/datasets/${encodeURIComponent(activeDataset)}/reindex`, { method: "POST" });
    }

    async function genRdbms() {
      if (!activeDataset) return;
      await runDatasetStream(`/api/datasets/${encodeURIComponent(activeDataset)}/rdbms/generate`, { method: "POST" });
    }

    async function summarizeDataset() {
      if (!activeDataset || busy) return;

      const prompt = `Summarize all documents in ${activeDataset}`;
      const chat = newChat();
      chat.name = `Summary: ${activeDataset}`;
      chat.messages = [{ role: "user", content: prompt }, { role: "assistant", content: "Loading cached summary..." }];
      try {
        const data = await api(`/api/datasets/${encodeURIComponent(activeDataset)}/summary`);
        chat.messages[1].content = data.summary || "No cached summary found.";
        setStats([{ label: "Mode", value: "Cached dataset summary" }]);
        addLog(`Loaded cached summary for ${activeDataset}.`);
      } catch (err) {
        chat.messages[1].content = err.message;
        addLog(`Summary unavailable: ${err.message}`);
      }
      setChats((items) => [...items, chat]);
      setCurrentChatId(chat.id);
    }

    async function discover() {
      if (!activeDataset || !discoverTopic.trim()) return;
      await runDatasetStream("/api/discover", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          topic: discoverTopic,
          dataset: activeDataset,
          max_papers: Number(discoverMax),
          num_queries: Number(discoverQueries),
        }),
      });
    }

    async function sendMessage(event) {
      event && event.preventDefault();
      const message = input.trim();
      if (!message || busy) return;

      const previousMessages = currentChat.messages;
      const renamed = previousMessages.length === 0 ? chatName(message) : currentChat.name;
      patchCurrentChat((chat) => ({
        ...chat,
        name: renamed,
        messages: [...chat.messages, { role: "user", content: message }, { role: "assistant", content: "" }],
      }));
      setInput("");
      setStats([]);
      setBusy(true);

      try {
        const res = await fetch("/api/chat/stream", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            message,
            mode,
            dataset: activeDataset,
            history: previousMessages.map((msg) => ({
              role: msg.role,
              content: textValue(msg.content),
            })),
          }),
        });
        let answer = "";
        await readEventStream(res, {
          token: (data) => {
            answer += textValue(data.delta);
            patchCurrentChat((chat) => ({
              ...chat,
              messages: chat.messages.map((msg, idx) =>
                idx === chat.messages.length - 1 ? { ...msg, role: "assistant", content: answer } : msg,
              ),
            }));
          },
          sources: (data) => {
            patchCurrentChat((chat) => ({
              ...chat,
              messages: chat.messages.map((msg, idx) =>
                idx === chat.messages.length - 1 ? { ...msg, role: "assistant", sources: data.sources || [] } : msg,
              ),
            }));
          },
          stats: (data) => setStats(data.stats || []),
          error: (data) => {
            answer = textValue(data.message) || "Unknown error.";
            patchCurrentChat((chat) => ({
              ...chat,
              messages: chat.messages.map((msg, idx) =>
                idx === chat.messages.length - 1 ? { role: "assistant", content: answer } : msg,
              ),
            }));
          },
          done: (data) => {
            setStats(data.stats || []);
            if (data.sources && data.sources.length) {
              patchCurrentChat((chat) => ({
                ...chat,
                messages: chat.messages.map((msg, idx) =>
                  idx === chat.messages.length - 1 ? { role: "assistant", content: answer, sources: data.sources } : msg,
                ),
              }));
            }
          },
        });
      } catch (err) {
        patchCurrentChat((chat) => ({
          ...chat,
          messages: chat.messages.map((msg, idx) =>
            idx === chat.messages.length - 1 ? { role: "assistant", content: err.message } : msg,
          ),
        }));
      } finally {
        setBusy(false);
      }
    }

    function startNewChat() {
      const chat = newChat();
      setChats((items) => [...items, chat]);
      setCurrentChatId(chat.id);
      setStats([]);
    }

    function deleteChat(chatId) {
      if (!confirm("Are you sure you want to remove this?")) return;
      if (chats.length === 1) {
        const chat = newChat();
        setChats([chat]);
        setCurrentChatId(chat.id);
        return;
      }
      const targetId = chatId || currentChat.id;
      const remaining = chats.filter((chat) => chat.id !== targetId);
      setChats(remaining);
      if (currentChatId === targetId) {
        setCurrentChatId(remaining[0].id);
      }
    }

    return h("div", { className: "app-shell" },
      h(Header, { boot, busy }),
      h("main", { className: "workspace" },
        h(ChatSidebar, {
          chats,
          currentChatId,
          setCurrentChatId,
          startNewChat,
          deleteChat,
          summarizeDataset,
          activeDataset,
          datasetDetail,
          busy,
        }),
        h("section", { className: "chat-panel" },
          h("div", { className: "mode-row" },
            MODES.map((item) =>
              h("button", {
                key: item,
                className: item === mode ? "mode active" : "mode",
                onClick: () => setMode(item),
              }, item),
            ),
          ),
          h(MessageList, { messages: currentChat.messages, busy, chatEndRef, activeDataset }),
          h("form", { className: "composer", onSubmit: sendMessage },
            h("textarea", {
              value: input,
              placeholder: mode === "Model Only"
                ? "Ask the model directly..."
                : mode === "SQL RDBMS"
                  ? "Ask an exact SQL-backed question about this dataset..."
                  : "Ask a question about your documents...",
              onChange: (event) => setInput(event.target.value),
              onKeyDown: (event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  sendMessage(event);
                }
              },
              rows: 1,
            }),
            h("button", { className: "send", disabled: busy || !input.trim() }, busy ? "Running" : "Send"),
          ),
          h(StatsPanel, { stats }),
        ),
        h(DatasetPanel, {
          datasets,
          activeDataset,
          datasetDetail,
          selectDataset,
          newDatasetName,
          setNewDatasetName,
          createDataset,
          deleteDataset,
          uploadFiles,
          reindex,
          genRdbms,
          discoverTopic,
          setDiscoverTopic,
          discoverMax,
          setDiscoverMax,
          discoverQueries,
          setDiscoverQueries,
          discover,
          log,
          busy,
        }),
      ),
    );
  }

  function Header({ boot, busy }) {
    return h("header", { className: "topbar" },
      h("img", { src: boot && boot.logo ? boot.logo : "/static/mylabs-logo.png", className: "brand-logo", alt: "MyLabs" }),
      h("div", null,
        h("div", { className: "brand-title" }, "MyLabs Studio"),
        h("div", { className: "brand-subtitle" }, "Nemotron-3-Nano RAG · React SPA"),
      ),
      h("div", { className: "topbar-spacer" }),
      h("div", { className: "chip" }, boot ? `${Math.round(boot.ctx_size / 1024)}K ctx` : "loading"),
      h("div", { className: busy ? "status busy" : "status" }, busy ? "Working" : "Online"),
    );
  }

  function ChatSidebar({ chats, currentChatId, setCurrentChatId, startNewChat, deleteChat, summarizeDataset, activeDataset, datasetDetail, busy }) {
    return h("aside", { className: "sidebar left" },
      h("button", {
        className: "wide",
        onClick: summarizeDataset,
        disabled: busy || !activeDataset || !datasetDetail || !datasetDetail.summary_available,
      }, "Summarize Dataset"),
      h("button", { className: "primary wide", onClick: startNewChat }, "+ New Chat"),
      h("div", { className: "section-label" }, "Conversations"),
      h("div", { className: "chat-list" },
        chats.map((chat, idx) =>
          h("div", { key: chat.id, className: chat.id === currentChatId ? "chat-row active" : "chat-row" },
            h("button", {
              className: "chat-tab",
              onClick: () => setCurrentChatId(chat.id),
            }, `${idx + 1}. ${chat.name}`),
            h("button", {
              className: "chat-remove",
              title: "Remove conversation",
              onClick: (event) => {
                event.stopPropagation();
                deleteChat(chat.id);
              },
            }, "×"),
          ),
        ),
      ),
    );
  }

  function MessageList({ messages, busy, chatEndRef, activeDataset }) {
    if (!messages.length) {
      return h("div", { className: "empty-state" },
        h("div", { className: "orb" }),
        h("h1", null, "Ask across your research corpus."),
        h("p", null, "Choose a dataset, pick a retrieval mode, and stream answers with visible pipeline timing."),
      );
    }
    return h("div", { className: "messages" },
      messages.map((msg, idx) =>
        h("article", { key: idx, className: `message ${msg.role}` },
          h("div", { className: "avatar" }, msg.role === "user" ? "You" : "AI"),
          h("div", { className: "bubble" },
            h(MessageText, {
              content: msg.content || (busy && msg.role === "assistant" ? "Thinking..." : ""),
              sources: msg.sources || [],
              dataset: activeDataset,
            }),
          ),
        ),
      ),
      h("div", { ref: chatEndRef }),
    );
  }

  function MessageText({ content, sources, dataset }) {
    const rendered = linkifyCitations(textValue(content), sources, dataset);
    const hasInlineLinks = rendered.some((part) => part && part.type === "a");
    return h("div", { className: "message-text" },
      rendered,
      sources.length && !hasInlineLinks ? h(InlineSourceList, { sources }) : null,
    );
  }

  function InlineSourceList({ sources }) {
    return h("span", { className: "inline-sources" },
      "\n\nSources: ",
      sources.map((source, idx) => [
        idx ? ", " : "",
        h("a", {
          key: `${source.url || source.label}-${idx}`,
          href: sourceUrlFromCitation("", source.source || source.label, source.page, source.chunk_index) || source.url || "#",
          target: "_blank",
          rel: "noopener noreferrer",
        }, source.label || String(source)),
      ]),
    );
  }

  function normalizeSourceName(value) {
    return String(value || "").trim().split("/").pop().toLowerCase();
  }

  function findSourceLink(sources, sourceName, page, chunkIndex) {
    const normalized = normalizeSourceName(sourceName);
    const pageNumber = page ? Number(page) : null;
    const chunkNumber = chunkIndex ? Number(chunkIndex) : null;

    return sources.find((source) => {
      if (normalizeSourceName(source.source || source.label) !== normalized) return false;
      if (pageNumber && Number(source.page || 0) !== pageNumber) return false;
      if (chunkNumber && source.chunk_index != null && Number(source.chunk_index) !== chunkNumber) return false;
      return true;
    }) || sources.find((source) => {
      if (normalizeSourceName(source.source || source.label) !== normalized) return false;
      if (pageNumber && Number(source.page || 0) !== pageNumber) return false;
      return true;
    }) || sources.find((source) => normalizeSourceName(source.source || source.label) === normalized);
  }

  function sourceUrlFromCitation(dataset, sourceName, page, chunkIndex) {
    if (!sourceName) return "";
    const sourcePart = encodeURIComponent(String(sourceName).trim().split("/").pop());
    const chunkQuery = chunkIndex ? `?chunk=${Number(chunkIndex)}` : "";
    if (page) {
      return `/api/documents/${sourcePart}/pages/${Number(page)}${chunkQuery}`;
    }
    return `/api/documents/${sourcePart}/text${chunkQuery}`;
  }

  function renderCitationLink(fullText, source, key, dataset, sourceName, page, chunkIndex) {
    const url = sourceUrlFromCitation(dataset, sourceName || (source && source.source), page || (source && source.page), chunkIndex || (source && source.chunk_index))
      || (source && source.url);
    if (!url) return fullText;
    return h("a", {
      className: "inline-source-link",
      href: url,
      target: "_blank",
      rel: "noopener noreferrer",
      key,
    }, fullText);
  }

  function linkifyCitations(content, sources, dataset) {
    if (!content) return [];

    const parts = [];
    const citationPattern = /\[(?:source|Source):\s*([^\]\|,]+)(?:\s*(?:,|\|)\s*(?:p\.?|page|Page)\s*\.?\s*(\d+))?(?:\s*(?:,|\|)\s*(?:chunk|Chunk)\s*(\d+))?[^\]]*\]|\[([^\]\|,]+\.pdf)(?:\s*(?:,|\|)\s*(?:p\.?|page|Page)\s*\.?\s*(\d+))?(?:\s*(?:,|\|)\s*(?:chunk|Chunk)\s*(\d+))?[^\]]*\]/g;
    let lastIndex = 0;
    let match;

    while ((match = citationPattern.exec(content)) !== null) {
      const fullText = match[0];
      const sourceName = match[1] || match[4];
      const page = match[2] || match[5];
      const chunkIndex = match[3] || match[6];
      if (match.index > lastIndex) {
        parts.push(content.slice(lastIndex, match.index));
      }

      const source = findSourceLink(sources, sourceName, page, chunkIndex);
      parts.push(renderCitationLink(fullText, source, `${sourceName}-${match.index}`, dataset, sourceName, page, chunkIndex));
      lastIndex = match.index + fullText.length;
    }

    if (lastIndex < content.length) {
      parts.push(content.slice(lastIndex));
    }

    return parts.flatMap((part, idx) => typeof part === "string" ? linkifySourceUrls(part, idx) : part);
  }

  function linkifySourceUrls(text, groupIndex) {
    const parts = [];
    const urlPattern = /(\/api\/documents\/[^\s|)]+(?:\?[^\s|)]+)?)/g;
    let lastIndex = 0;
    let match;

    while ((match = urlPattern.exec(text)) !== null) {
      if (match.index > lastIndex) {
        parts.push(text.slice(lastIndex, match.index));
      }
      const url = match[1];
      parts.push(h("a", {
        className: "inline-source-link",
        href: url,
        target: "_blank",
        rel: "noopener noreferrer",
        key: `url-${groupIndex}-${match.index}`,
      }, url));
      lastIndex = match.index + url.length;
    }

    if (lastIndex < text.length) {
      parts.push(text.slice(lastIndex));
    }
    return parts;
  }

  function StatsPanel({ stats }) {
    return h("div", { className: "stats-card" },
      h("div", { className: "section-label" }, "Pipeline Stats"),
      stats.length
        ? stats.map((item, idx) =>
            h("div", { className: "stat-row", key: idx },
              h("span", null, item.label),
              h("strong", null, item.value),
            ),
          )
        : h("p", { className: "muted" }, "Stats appear while a response is streaming."),
    );
  }

  function DatasetPanel(props) {
    return h("aside", { className: "sidebar right" },
      h("div", { className: "section-label" }, "Dataset"),
      h("select", {
        value: props.activeDataset || "",
        onChange: (event) => props.selectDataset(event.target.value),
      },
        h("option", { value: "", disabled: true }, "Select dataset"),
        props.datasets.map((dataset) =>
          h("option", { key: dataset.name, value: dataset.name },
            `${dataset.name} · ${dataset.chunks || 0} chunks`,
          ),
        ),
      ),
      h("div", { className: "dataset-card" },
        props.datasetDetail
          ? [
              h("strong", { key: "name" }, props.datasetDetail.name),
              h("span", { key: "meta" }, `${props.datasetDetail.file_count} files · ${props.datasetDetail.chunks} chunks`),
              h("span", {
                key: "rdbms",
                className: props.datasetDetail.rdbms_available ? "rdbms-status ready" : "rdbms-status",
              }, props.datasetDetail.rdbms_available
                ? `RDBMS ready · ${props.datasetDetail.rdbms_domain || "general"}`
                : "RDBMS not generated"),
              h("div", { className: "file-list", key: "files" },
                props.datasetDetail.files.length
                  ? props.datasetDetail.files.map((file) => h("small", { key: file }, file))
                  : h("small", null, "No files yet."),
              ),
            ]
          : h("span", null, "No dataset selected."),
      ),
      h("div", { className: "split" },
        h("input", {
          value: props.newDatasetName,
          placeholder: "new_dataset",
          onChange: (event) => props.setNewDatasetName(event.target.value),
        }),
        h("button", { className: "primary", onClick: props.createDataset }, "Create"),
      ),
      h("label", { className: props.busy ? "upload disabled" : "upload" },
        "Upload & Ingest",
        h("input", { type: "file", multiple: true, onChange: props.uploadFiles, disabled: props.busy || !props.activeDataset }),
      ),
      h("div", { className: "split" },
        h("button", { onClick: props.reindex, disabled: props.busy || !props.activeDataset }, "Re-index"),
        h("button", { onClick: props.genRdbms, disabled: props.busy || !props.activeDataset }, "Gen RDBMS"),
      ),
      h("button", { className: "danger wide bottom", onClick: props.deleteDataset, disabled: props.busy || !props.activeDataset }, "Delete"),
      h("div", { className: "section-label" }, "Discover Papers"),
      h("input", {
        value: props.discoverTopic,
        placeholder: "particle physics Higgs boson",
        onChange: (event) => props.setDiscoverTopic(event.target.value),
      }),
      h("div", { className: "split" },
        h("label", null, "Max",
          h("input", {
            type: "number",
            min: 5,
            max: 50,
            value: props.discoverMax,
            onChange: (event) => props.setDiscoverMax(event.target.value),
          }),
        ),
        h("label", null, "Queries",
          h("input", {
            type: "number",
            min: 2,
            max: 12,
            value: props.discoverQueries,
            onChange: (event) => props.setDiscoverQueries(event.target.value),
          }),
        ),
      ),
      h("button", { className: "primary wide", onClick: props.discover, disabled: props.busy || !props.activeDataset }, "Search & Download"),
      h("div", { className: "section-label" }, "Activity"),
      h("div", { className: "log" },
        props.log.length ? props.log.map((item, idx) => h("div", { key: idx }, item)) : h("span", null, "No activity yet."),
      ),
    );
  }

  ReactDOM.createRoot(document.getElementById("root")).render(h(App));
})();
