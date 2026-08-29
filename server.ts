import express from "express";
import path from "path";
import { fileURLToPath } from "url";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI, GenerateVideosOperation } from "@google/genai";
import dotenv from "dotenv";

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = 3000;

// System instructions fallback
const PERSONA_PROMPTS: Record<string, string> = {
  tech_expert: `You are a world-class Senior Software Engineer and Technical Architect.
Your role:
- Analyze technical problems with depth, clarity, and precision.
- Provide idiomatic, clean, modular, and production-ready code examples in appropriate languages with markdown formatting.
- Explain trade-offs, time and space complexity, edge cases, security considerations, and debugging strategies.
- Maintain an insightful, professional, and knowledgeable tone.`,

  friendly_teacher: `You are a warm, encouraging, and intuitive Educator.
Your role:
- Explain complex concepts simply using relatable real-world analogies, step-by-step breakdowns, and intuitive examples.
- Structure explanations progressively from fundamentals to deeper insights.
- Use an inviting, enthusiastic, and supportive tone that makes learning fun and engaging.
- Provide quick comprehension checks or thought experiments where helpful.`,

  personal_assistant: `You are an elite Executive Personal Assistant and Productivity Specialist.
Your role:
- Deliver concise, highly organized, and immediately actionable answers.
- Structure outputs using clear bullet points, bold headers, formatted checklists, or executive summaries.
- Focus on high efficiency, time-saving workflows, practical prioritization, and zero fluff.
- Maintain a proactive, polished, and structured tone.`
};

let genAIClient: GoogleGenAI | null = null;

function getGenAI(): GoogleGenAI {
  if (!genAIClient) {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      console.warn("WARNING: GEMINI_API_KEY environment variable is not set. Using empty fallback for development.");
    }
    genAIClient = new GoogleGenAI({
      apiKey: apiKey || "",
      httpOptions: {
        headers: {
          "User-Agent": "aistudio-build",
        },
      },
    });
  }
  return genAIClient;
}

async function startServer() {
  const app = express();
  app.use(express.json({ limit: "10mb" }));

  // Health check endpoint
  app.get("/api/health", (_req, res) => {
    res.json({
      status: "ok",
      hasApiKey: !!process.env.GEMINI_API_KEY,
      timestamp: Date.now()
    });
  });

  // Chat completion endpoint
  app.post("/api/chat", async (req, res) => {
    try {
      const { messages, personaId, systemPrompt, temperature } = req.body;

      if (!messages || !Array.isArray(messages) || messages.length === 0) {
        return res.status(400).json({ error: "Messages array is required." });
      }

      const activeSystemInstruction =
        systemPrompt ||
        PERSONA_PROMPTS[personaId] ||
        PERSONA_PROMPTS.tech_expert;

      const ai = getGenAI();

      // Format conversation history for Gemini API
      // Transform incoming messages into contents format with multimodal image support
      const contents = messages
        .filter((msg: { sender: string; content?: string; imageUrl?: string }) => 
          (msg.content || msg.imageUrl) && (msg.sender === "user" || msg.sender === "assistant")
        )
        .map((msg: { sender: string; content?: string; imageUrl?: string; imageMimeType?: string }) => {
          const parts: any[] = [];

          // If message contains an attached image (data:image/...;base64,...)
          if (msg.imageUrl) {
            const dataUrlMatch = msg.imageUrl.match(/^data:([^;]+);base64,(.+)$/);
            if (dataUrlMatch) {
              const mimeType = dataUrlMatch[1] || msg.imageMimeType || "image/jpeg";
              const base64Data = dataUrlMatch[2];
              parts.push({
                inlineData: {
                  mimeType: mimeType,
                  data: base64Data,
                },
              });
            } else if (msg.imageUrl.startsWith("http")) {
              // Remote URL fallback or text reference
              parts.push({ text: `[Attached Image URL: ${msg.imageUrl}]` });
            }
          }

          if (msg.content && msg.content.trim()) {
            parts.push({ text: msg.content.trim() });
          } else if (parts.length === 0) {
            parts.push({ text: "Please analyze this attached image." });
          }

          return {
            role: msg.sender === "user" ? "user" : "model",
            parts,
          };
        });

      if (contents.length === 0) {
        return res.status(400).json({ error: "No valid user or assistant messages provided." });
      }

      const response = await ai.models.generateContent({
        model: "gemini-3.7-flash",
        contents,
        config: {
          systemInstruction: activeSystemInstruction,
          temperature: typeof temperature === "number" ? temperature : 0.7,
        },
      });

      const responseText = response.text || "I was unable to generate a response. Please try again.";

      res.json({
        content: responseText,
        personaId: personaId || "tech_expert",
        model: "gemini-3.7-flash",
        timestamp: Date.now(),
      });
    } catch (error: any) {
      console.error("Gemini API Error:", error);
      res.status(500).json({
        error: error.message || "Failed to process chat request.",
        details: process.env.NODE_ENV !== "production" ? String(error) : undefined,
      });
    }
  });

  // Prompt Enhancement for Veo Video Generation
  app.post("/api/enhance-video-prompt", async (req, res) => {
    try {
      const { prompt, stylePreset, aspectRatio } = req.body;
      if (!prompt || !prompt.trim()) {
        return res.status(400).json({ error: "Prompt is required." });
      }

      const ai = getGenAI();
      const styleInstruction = stylePreset ? `Visual style: ${stylePreset}.` : "";
      const aspectInstruction = aspectRatio ? `Target aspect ratio: ${aspectRatio}.` : "";

      const response = await ai.models.generateContent({
        model: "gemini-3.7-flash",
        contents: `You are an expert video director and AI cinematographer for Veo video generation.
Enhance the following basic prompt into a concise, vivid, and highly descriptive video prompt (under 60 words).
Include camera movement (e.g. slow pan, drone flythrough, dolly zoom, orbital tracking), lighting (e.g. golden hour volumetric rays, neon reflections, cinematic rim light), physical action/motion dynamics, texture details, and depth of field.
Do not include quotation marks or meta-commentary; return ONLY the enhanced video prompt.

User idea: "${prompt.trim()}"
${styleInstruction}
${aspectInstruction}`,
        config: {
          temperature: 0.7,
        },
      });

      const enhancedPrompt = response.text?.trim() || prompt.trim();
      res.json({ enhancedPrompt });
    } catch (error: any) {
      console.error("Video prompt enhance error:", error);
      res.json({ enhancedPrompt: req.body.prompt || "" });
    }
  });

  // Start Veo Video Generation (3-Step Pattern)
  app.post("/api/generate-video", async (req, res) => {
    try {
      const { prompt, aspectRatio = "16:9", resolution = "720p" } = req.body;
      if (!prompt || !prompt.trim()) {
        return res.status(400).json({ error: "Prompt is required for video generation." });
      }

      const ai = getGenAI();
      const validAspect = aspectRatio === "9:16" ? "9:16" : "16:9";
      const validRes = resolution === "1080p" ? "1080p" : "720p";

      // Select model according to @google/genai guidelines: veo-3.1-lite-generate-preview
      const operation = await ai.models.generateVideos({
        model: "veo-3.1-lite-generate-preview",
        prompt: prompt.trim(),
        config: {
          numberOfVideos: 1,
          resolution: validRes,
          aspectRatio: validAspect,
        },
      });

      res.json({
        operationName: operation.name,
        aspectRatio: validAspect,
        resolution: validRes,
        prompt: prompt.trim(),
        status: "processing",
      });
    } catch (error: any) {
      console.error("Veo Generate Video Error:", error);
      res.status(500).json({
        error: error.message || "Failed to start video generation.",
        details: process.env.NODE_ENV !== "production" ? String(error) : undefined,
      });
    }
  });

  // Check Veo Video Generation Status
  app.post("/api/video-status", async (req, res) => {
    try {
      const { operationName } = req.body;
      if (!operationName) {
        return res.status(400).json({ error: "operationName is required." });
      }

      const ai = getGenAI();
      const op = new GenerateVideosOperation();
      op.name = operationName;

      const updated = await ai.operations.getVideosOperation({ operation: op });
      const isDone = Boolean(updated.done);
      const videoUri = updated.response?.generatedVideos?.[0]?.video?.uri;
      const error = updated.error;

      res.json({
        done: isDone,
        hasVideoUri: Boolean(videoUri),
        error: error ? String(error) : null,
      });
    } catch (error: any) {
      console.error("Veo Video Status Error:", error);
      res.status(500).json({
        error: error.message || "Failed to check video status.",
      });
    }
  });

  // Download / Stream Generated Video
  app.post("/api/video-download", async (req, res) => {
    try {
      const { operationName } = req.body;
      if (!operationName) {
        return res.status(400).json({ error: "operationName is required." });
      }

      const apiKey = process.env.GEMINI_API_KEY;
      if (!apiKey) {
        return res.status(400).json({ error: "GEMINI_API_KEY is not configured on server." });
      }

      const ai = getGenAI();
      const op = new GenerateVideosOperation();
      op.name = operationName;

      const updated = await ai.operations.getVideosOperation({ operation: op });
      const uri = updated.response?.generatedVideos?.[0]?.video?.uri;

      if (!uri) {
        return res.status(404).json({ error: "Generated video URI not ready or not found." });
      }

      const videoRes = await fetch(uri, {
        headers: { "x-goog-api-key": apiKey },
      });

      if (!videoRes.ok) {
        return res.status(videoRes.status).json({ error: "Failed to download video stream from Google." });
      }

      res.setHeader("Content-Type", "video/mp4");
      res.setHeader("Content-Disposition", `inline; filename="veo-generated-video.mp4"`);

      if (videoRes.body) {
        const reader = videoRes.body.getReader();
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          res.write(value);
        }
        res.end();
      } else {
        res.end();
      }
    } catch (error: any) {
      console.error("Veo Video Download Error:", error);
      res.status(500).json({
        error: error.message || "Failed to stream generated video.",
      });
    }
  });

  // Vite development middleware or static production handler
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (_req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Android Gemini Chat server running on http://localhost:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error("Failed to start server:", err);
  process.exit(1);
});
