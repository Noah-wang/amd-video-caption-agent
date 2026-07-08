import OpenAI from "openai";

const MODEL =
  process.env.FIREWORKS_VISION_MODEL || "accounts/fireworks/models/minimax-m3";

function extractJson(text) {
  const trimmed = text.trim();
  const fenced = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/i);
  const candidate = fenced ? fenced[1].trim() : trimmed;
  const start = candidate.indexOf("{");
  const end = candidate.lastIndexOf("}");

  if (start === -1 || end === -1 || end <= start) {
    throw new Error("Model response did not contain a JSON object.");
  }

  return JSON.parse(candidate.slice(start, end + 1));
}

function normalizeCaption(value) {
  return String(value || "")
    .replace(/\s+/g, " ")
    .trim();
}

export default async function handler(request, response) {
  if (request.method !== "POST") {
    response.setHeader("Allow", "POST");
    return response.status(405).json({ error: "Method not allowed" });
  }

  if (!process.env.FIREWORKS_API_KEY) {
    return response.status(500).json({
      error: "FIREWORKS_API_KEY is not configured on the server.",
    });
  }

  const { contactSheet, videoUrl } = request.body || {};

  if (!contactSheet || !String(contactSheet).startsWith("data:image/")) {
    return response.status(400).json({
      error: "Missing contactSheet data URL.",
    });
  }

  const client = new OpenAI({
    apiKey: process.env.FIREWORKS_API_KEY,
    baseURL: "https://api.fireworks.ai/inference/v1",
  });

  const prompt = `You are generating captions for an AI hackathon video captioning demo.

Look at the contact sheet. It contains representative frames from one short video.

Return ONLY valid JSON with this exact shape:
{
  "visual_facts": "one concise factual summary of the video",
  "captions": {
    "formal": "8 to 20 words, objective and professional",
    "sarcastic": "8 to 20 words, dry and lightly mocking",
    "humorous_tech": "8 to 20 words, funny with a clear tech reference",
    "humorous_non_tech": "8 to 20 words, funny with no tech jargon"
  }
}

Do not mention the contact sheet, frames, image, model, or prompt. Stay grounded in visible evidence.
Video URL, if useful for context: ${videoUrl || "not provided"}`;

  try {
    const completion = await client.chat.completions.create({
      model: MODEL,
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: prompt },
            { type: "image_url", image_url: { url: contactSheet } },
          ],
        },
      ],
      max_tokens: 520,
      temperature: 0.72,
    });

    const content = completion.choices?.[0]?.message?.content || "";
    const parsed = extractJson(content);
    const captions = parsed.captions || {};

    return response.status(200).json({
      model: MODEL,
      visual_facts: normalizeCaption(parsed.visual_facts),
      captions: {
        formal: normalizeCaption(captions.formal),
        sarcastic: normalizeCaption(captions.sarcastic),
        humorous_tech: normalizeCaption(captions.humorous_tech),
        humorous_non_tech: normalizeCaption(captions.humorous_non_tech),
      },
    });
  } catch (error) {
    return response.status(500).json({
      error: error.message || "Caption generation failed.",
    });
  }
}
