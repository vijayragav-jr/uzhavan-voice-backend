import dns from 'node:dns';
dns.setDefaultResultOrder('ipv4first');

import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import { GoogleGenAI, Type } from '@google/genai';

const app = express();
const PORT = Number(process.env.PORT) || 3000;
const PRIMARY_MODEL = 'gemini-3.8-flash';
const REQUEST_TIMEOUT_MS = Number(process.env.GEMINI_TIMEOUT_MS) || 25000;

app.use(cors());
app.use(express.json({ limit: '25mb' }));

function getGeminiApiKey() {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey || !apiKey.trim()) {
    throw new Error('GEMINI_API_KEY is not configured in .env');
  }
  return apiKey.trim();
}

function getGeminiClient() {
  return new GoogleGenAI({ apiKey: getGeminiApiKey() });
}

function isRetryableOverloadOrTimeout(err) {
  if (!err) return false;
  const status = err.status ?? err.code ?? err?.error?.code;
  if (
    status === 503 ||
    status === 429 ||
    status === 'ETIMEDOUT' ||
    status === 'ABORT_ERR'
  ) {
    return true;
  }
  const msg = String(err.message || err).toLowerCase();
  return (
    msg.includes('503') ||
    msg.includes('unavailable') ||
    msg.includes('overloaded') ||
    msg.includes('high demand') ||
    msg.includes('timeout') ||
    msg.includes('timed out') ||
    msg.includes('aborted')
  );
}

const DEPRECATED_MODEL_PREFIXES = ['gemini-2.5', 'gemini-2.0', 'gemini-1.5'];
const EXCLUDED_CAPABILITY_SUBSTRINGS = ['embedding', 'imagen', 'veo', 'tts'];

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function listAvailableFallbackModels(excludeModel) {
  const apiKey = getGeminiApiKey();
  const url = `https://generativelanguage.googleapis.com/v1beta/models?key=${encodeURIComponent(apiKey)}`;
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to list Gemini models (${res.status}): ${await res.text()}`);
  }
  const data = await res.json();
  const models = Array.isArray(data.models) ? data.models : [];

  return models
    .filter((m) => {
      const methods = Array.isArray(m.supportedGenerationMethods)
        ? m.supportedGenerationMethods
        : [];
      if (!methods.includes('generateContent')) return false;

      const id = String(m.name || '').replace(/^models\//, '').toLowerCase();
      if (!id || id === excludeModel.toLowerCase()) return false;
      if (DEPRECATED_MODEL_PREFIXES.some((prefix) => id.startsWith(prefix))) {
        return false;
      }
      if (EXCLUDED_CAPABILITY_SUBSTRINGS.some((ex) => id.includes(ex))) {
        return false;
      }
      return id.startsWith('gemini-3') || id.startsWith('gemma-3');
    })
    .map((m) => String(m.name).replace(/^models\//, ''))
    .sort((a, b) => {
      const aGemini = a.toLowerCase().startsWith('gemini-3') ? 0 : 1;
      const bGemini = b.toLowerCase().startsWith('gemini-3') ? 0 : 1;
      if (aGemini !== bGemini) return aGemini - bGemini;
      const aFlash = a.toLowerCase().includes('flash') ? 0 : 1;
      const bFlash = b.toLowerCase().includes('flash') ? 0 : 1;
      return aFlash - bFlash;
    });
}

async function callWithTimeout(ai, params, model) {
  return Promise.race([
    ai.models.generateContent({
      ...params,
      model,
    }),
    new Promise((_, reject) =>
      setTimeout(
        () =>
          reject(
            new Error(`Gemini request timed out after ${REQUEST_TIMEOUT_MS}ms`)
          ),
        REQUEST_TIMEOUT_MS
      )
    ),
  ]);
}

async function generateWithFallback(params) {
  const ai = getGeminiClient();

  try {
    const response = await callWithTimeout(ai, params, PRIMARY_MODEL);
    return { response, model: PRIMARY_MODEL };
  } catch (firstErr) {
    if (!isRetryableOverloadOrTimeout(firstErr)) {
      throw firstErr;
    }

    console.warn(
      `[Gemini Fallback] ${PRIMARY_MODEL} attempt 1 failed (${firstErr?.message || firstErr}); retrying ${PRIMARY_MODEL} after 1500ms...`
    );
    await sleep(1500);

    try {
      const response = await callWithTimeout(ai, params, PRIMARY_MODEL);
      return { response, model: PRIMARY_MODEL };
    } catch (secondErr) {
      if (!isRetryableOverloadOrTimeout(secondErr)) {
        throw secondErr;
      }

      console.warn(
        `[Gemini Fallback] ${PRIMARY_MODEL} attempt 2 failed (${secondErr?.message || secondErr}); fetching active gemini-3 / gemma-3 fallback models...`
      );

      const fallbackModels = await listAvailableFallbackModels(PRIMARY_MODEL);
      if (fallbackModels.length === 0) {
        throw secondErr;
      }

      console.warn(
        `[Gemini Fallback] Candidate fallback models: ${fallbackModels.join(', ')}`
      );

      let lastError = secondErr;
      for (const model of fallbackModels) {
        try {
          const response = await callWithTimeout(ai, params, model);
          return { response, model };
        } catch (err) {
          lastError = err;
          if (isRetryableOverloadOrTimeout(err)) {
            console.warn(
              `[Gemini Fallback] ${model} failed (${err?.message || err}); trying next model...`
            );
            continue;
          }
          throw err;
        }
      }

      throw lastError;
    }
  }
}

const VALID_INTENTS = [
  'OPEN_HOME',
  'OPEN_SCANNER',
  'OPEN_PLANS',
  'OPEN_MENU',
  'OPEN_CALENDAR',
  'OPEN_FERTILIZER',
  'OPEN_PEST_LIBRARY',
  'OPEN_WEATHER',
  'OPEN_CROP_TIPS',
  'OPEN_SOIL_SENSOR',
  'CHECK_SOIL_MOISTURE',
  'GO_BACK',
  'START_GUIDED_PLAN',
  'NEXT_STEP',
  'PREVIOUS_STEP',
  'MARK_STEP_DONE',
  'CHANGE_LANGUAGE',
  'ANSWER_QUESTION',
  'UNKNOWN',
];

const VALID_LANGUAGES = ['ta-IN', 'te-IN', 'en-IN'];
const VALID_CROPS = ['paddy', 'tomato', 'banana', 'cotton', 'groundnut', 'chilli'];

const SUPPORTED_AUDIO_MIME_MAP = {
  'audio/m4a': 'audio/mp4',
  'audio/x-m4a': 'audio/mp4',
  'audio/mp4': 'audio/mp4',
  'audio/aac': 'audio/aac',
  'audio/wav': 'audio/wav',
  'audio/x-wav': 'audio/wav',
};

const VOICE_INTENT_RESPONSE_SCHEMA = {
  type: Type.OBJECT,
  properties: {
    intent: {
      type: Type.STRING,
      enum: VALID_INTENTS,
      description: 'The single best matching navigation, tool, or Q&A intent.',
    },
    targetLanguage: {
      type: Type.STRING,
      nullable: true,
      enum: ['ta-IN', 'te-IN', 'en-IN'],
      description:
        'Set to ta-IN, te-IN, or en-IN ONLY when intent is CHANGE_LANGUAGE; otherwise null.',
    },
    targetCrop: {
      type: Type.STRING,
      nullable: true,
      enum: VALID_CROPS,
      description:
        'Optional crop mentioned by the farmer (paddy, tomato, banana, cotton, groundnut, chilli), otherwise null.',
    },
    acres: {
      type: Type.NUMBER,
      nullable: true,
      description:
        'Optional land area in acres mentioned for fertilizer calculation (e.g., 2.0), otherwise null.',
    },
    spokenReply: {
      type: Type.STRING,
      description:
        'Short, warm, farmer-friendly confirmation (or a clear 3-sentence practical answer when intent is ANSWER_QUESTION) in targetLanguage (if CHANGE_LANGUAGE) or activeLanguage.',
    },
  },
  required: ['intent', 'targetLanguage', 'spokenReply'],
};

const AGRONOMIST_QA_SCHEMA = {
  type: Type.OBJECT,
  properties: {
    answer: {
      type: Type.STRING,
      description:
        'Clear, practical 3-sentence farming answer in simple farmer-friendly activeLanguage.',
    },
    topic: {
      type: Type.STRING,
      description: 'Short 2-4 word topic title in activeLanguage.',
    },
    cropName: {
      type: Type.STRING,
      nullable: true,
      description: 'Related crop name if applicable, otherwise null.',
    },
  },
  required: ['answer', 'topic'],
};

const CROP_DIAGNOSIS_RESPONSE_SCHEMA = {
  type: Type.OBJECT,
  properties: {
    cropName: {
      type: Type.STRING,
      description: 'Identified crop name, localized or with common name.',
    },
    diseaseName: {
      type: Type.STRING,
      description:
        'Identified disease, pest, nutrient deficiency, or Healthy if no issue.',
    },
    severity: {
      type: Type.STRING,
      enum: ['Low', 'Moderate', 'High'],
      description: 'Severity level of the issue.',
    },
    voiceSummary: {
      type: Type.STRING,
      description:
        'A concise 2-sentence farmer-friendly spoken summary in the requested activeLanguage.',
    },
    treatmentSteps: {
      type: Type.ARRAY,
      description: 'Ordered day-by-day practical field treatment steps.',
      items: {
        type: Type.OBJECT,
        properties: {
          dayNumber: {
            type: Type.INTEGER,
            description: 'Day number starting at 1 (e.g. 1, 2, 4).',
          },
          title: {
            type: Type.STRING,
            description: 'Short action title in activeLanguage.',
          },
          instruction: {
            type: Type.STRING,
            description:
              'Clear, simple, farmer-friendly field instruction in activeLanguage.',
          },
        },
        required: ['dayNumber', 'title', 'instruction'],
      },
    },
  },
  required: [
    'cropName',
    'diseaseName',
    'severity',
    'voiceSummary',
    'treatmentSteps',
  ],
};

function languageLabel(code) {
  switch (code) {
    case 'ta-IN':
      return 'Tamil (ta-IN)';
    case 'te-IN':
      return 'Telugu (te-IN)';
    case 'en-IN':
    default:
      return 'English (en-IN)';
  }
}

app.get('/health', (_req, res) => {
  res.json({
    ok: true,
    primaryModel: PRIMARY_MODEL,
    hasApiKey: Boolean(process.env.GEMINI_API_KEY?.trim()),
  });
});

app.post('/api/voice-intent', async (req, res) => {
  try {
    const {
      audioBase64,
      mimeType = 'audio/m4a',
      activeLanguage = 'en-IN',
      currentScreen = 'home',
    } = req.body ?? {};

    if (!audioBase64 || typeof audioBase64 !== 'string') {
      return res.status(400).json({ error: 'audioBase64 is required' });
    }

    const audioBytes = Buffer.from(audioBase64, 'base64').length;
    const rawMime = String(mimeType).toLowerCase().trim();
    const geminiAudioMime = SUPPORTED_AUDIO_MIME_MAP[rawMime] || 'audio/mp4';
    const safeLang = VALID_LANGUAGES.includes(activeLanguage)
      ? activeLanguage
      : 'en-IN';

    console.log(
      `[POST /api/voice-intent] bytes=${audioBytes}, mimeType=${rawMime} (-> ${geminiAudioMime}), activeLanguage=${safeLang}, currentScreen=${currentScreen}`
    );

    const systemInstruction = [
      'You are the voice command router and crop advisor for UzhavanVoice (F-Vision), a trilingual farming assistant app.',
      'You understand Tamil (ta-IN), Telugu (te-IN), English (en-IN), and Tanglish/Tenglish code-mixed speech.',
      'Examples:',
      '- Navigation: "Menu open pannu" -> OPEN_MENU, "Scanner open cheyi" -> OPEN_SCANNER, "Next step sollu" -> NEXT_STEP.',
      '- Farm Tools: "Cultivation calendar open pannu" or "Paddy cultivation steps cheppu" -> OPEN_CALENDAR (targetCrop: paddy).',
      '- "Fertilizer calculator open pannu" or "2 acres paddy urea evlo podanum" -> OPEN_FERTILIZER (targetCrop: paddy, acres: 2).',
      '- "Pest library open cheyi" -> OPEN_PEST_LIBRARY.',
      '- "Spray weather alert sollu" or "Weather open pannu" -> OPEN_WEATHER.',
      '- "Crop tips sollu" or "Soil health open cheyi" -> OPEN_CROP_TIPS.',
      '- "Soil sensor open pannu" or "Soil sensor open cheyi" or "Open IoT sensor" -> OPEN_SOIL_SENSOR.',
      '- "Soil moisture evlo irukku" or "Soil moisture cheppu" or "Check soil moisture" or "Manni erapatham evlo" -> CHECK_SOIL_MOISTURE.',
      '- General farming questions (e.g., "How to control whitefly in tomato organically?" or "தக்காளி இலை சுருட்டல் நோய்க்கு என்ன மருந்து?"): return intent ANSWER_QUESTION with a clear, practical 3-sentence answer in spokenReply in activeLanguage.',
      `Current screen: ${currentScreen}.`,
      `Active app language: ${languageLabel(safeLang)}.`,
      'If the user asks to switch language, set intent to CHANGE_LANGUAGE and targetLanguage to ta-IN, te-IN, or en-IN, and write spokenReply in that new targetLanguage.',
      'For all other intents, set targetLanguage to null and write spokenReply in activeLanguage.',
      'If the audio is silent or unintelligible, return intent UNKNOWN with a helpful spokenReply in activeLanguage.',
    ].join(' ');

    const { response, model: usedModel } = await generateWithFallback({
      contents: [
        {
          role: 'user',
          parts: [
            {
              text: 'Classify the voice command or farming question in the attached audio and return structured JSON.',
            },
            {
              inlineData: {
                mimeType: geminiAudioMime,
                data: audioBase64,
              },
            },
          ],
        },
      ],
      config: {
        systemInstruction,
        responseMimeType: 'application/json',
        responseSchema: VOICE_INTENT_RESPONSE_SCHEMA,
        temperature: 0.15,
      },
    });

    const rawText = response.text || '{}';
    console.log(`[POST /api/voice-intent] (${usedModel}) Gemini raw JSON:`, rawText);

    const parsed = JSON.parse(rawText);

    const intent = VALID_INTENTS.includes(parsed.intent)
      ? parsed.intent
      : 'UNKNOWN';
    const targetLanguage =
      intent === 'CHANGE_LANGUAGE' &&
      VALID_LANGUAGES.includes(parsed.targetLanguage)
        ? parsed.targetLanguage
        : null;
    const targetCrop = VALID_CROPS.includes(parsed.targetCrop)
      ? parsed.targetCrop
      : null;
    const acres =
      typeof parsed.acres === 'number' && Number.isFinite(parsed.acres) && parsed.acres > 0
        ? parsed.acres
        : null;

    return res.json({
      intent,
      targetLanguage,
      targetCrop,
      acres,
      spokenReply:
        typeof parsed.spokenReply === 'string' ? parsed.spokenReply : '',
    });
  } catch (err) {
    console.error('Error in /api/voice-intent:', err);
    return res.status(500).json({
      error:
        err instanceof Error ? err.message : 'Voice intent processing failed',
    });
  }
});

app.post('/api/diagnose-leaf', async (req, res) => {
  try {
    const {
      imageBase64,
      mimeType = 'image/jpeg',
      activeLanguage = 'en-IN',
    } = req.body ?? {};

    if (!imageBase64 || typeof imageBase64 !== 'string') {
      return res.status(400).json({ error: 'imageBase64 is required' });
    }

    const imageBytes = Buffer.from(imageBase64, 'base64').length;
    const safeLang = VALID_LANGUAGES.includes(activeLanguage)
      ? activeLanguage
      : 'en-IN';

    console.log(
      `[POST /api/diagnose-leaf] bytes=${imageBytes}, mimeType=${mimeType}, activeLanguage=${safeLang}`
    );

    const systemInstruction = [
      'You are an expert agronomist and Plantix-style crop doctor.',
      'Analyze the attached crop leaf image carefully to identify the crop, any disease/pest/deficiency, and severity.',
      `Write cropName, diseaseName, voiceSummary, and all treatmentSteps (title and instruction) in simple, farmer-friendly ${languageLabel(safeLang)}.`,
      'Keep voiceSummary to 2 clear spoken sentences suitable for text-to-speech.',
      'Provide practical, safe, day-by-day treatmentSteps ordered by dayNumber (e.g., Day 1, Day 2, Day 4).',
    ].join(' ');

    const { response, model: usedModel } = await generateWithFallback({
      contents: [
        {
          role: 'user',
          parts: [
            {
              text: `Diagnose this crop leaf image and return structured JSON in ${languageLabel(safeLang)}.`,
            },
            {
              inlineData: {
                mimeType,
                data: imageBase64,
              },
            },
          ],
        },
      ],
      config: {
        systemInstruction,
        responseMimeType: 'application/json',
        responseSchema: CROP_DIAGNOSIS_RESPONSE_SCHEMA,
        temperature: 0.2,
      },
    });

    const rawText = response.text || '{}';
    console.log(`[POST /api/diagnose-leaf] (${usedModel}) Gemini raw JSON:`, rawText);

    const parsed = JSON.parse(rawText);

    return res.json({
      cropName: parsed.cropName || 'Unknown Crop',
      diseaseName: parsed.diseaseName || 'Observation Needed',
      severity: ['Low', 'Moderate', 'High'].includes(parsed.severity)
        ? parsed.severity
        : 'Moderate',
      voiceSummary: parsed.voiceSummary || '',
      languageCode: safeLang,
      treatmentSteps: Array.isArray(parsed.treatmentSteps)
        ? parsed.treatmentSteps
        : [],
    });
  } catch (err) {
    console.error('Error in /api/diagnose-leaf:', err);
    return res.status(500).json({
      error: err instanceof Error ? err.message : 'Leaf diagnosis failed',
    });
  }
});

app.post('/api/ask-agronomist', async (req, res) => {
  try {
    const {
      question,
      audioBase64,
      mimeType = 'audio/m4a',
      activeLanguage = 'en-IN',
    } = req.body ?? {};

    if ((!question || typeof question !== 'string') && (!audioBase64 || typeof audioBase64 !== 'string')) {
      return res.status(400).json({ error: 'question or audioBase64 is required' });
    }

    const safeLang = VALID_LANGUAGES.includes(activeLanguage)
      ? activeLanguage
      : 'en-IN';

    const parts = [];
    if (question && typeof question === 'string' && question.trim()) {
      parts.push({ text: question.trim() });
    } else {
      parts.push({
        text: `Listen to the farmer's question in the attached audio and answer clearly in ${languageLabel(safeLang)}.`,
      });
    }

    if (audioBase64 && typeof audioBase64 === 'string') {
      const rawMime = String(mimeType).toLowerCase().trim();
      const geminiAudioMime = SUPPORTED_AUDIO_MIME_MAP[rawMime] || 'audio/mp4';
      const audioBytes = Buffer.from(audioBase64, 'base64').length;
      console.log(
        `[POST /api/ask-agronomist] audioBytes=${audioBytes}, mimeType=${rawMime} (-> ${geminiAudioMime}), activeLanguage=${safeLang}`
      );
      parts.push({
        inlineData: {
          mimeType: geminiAudioMime,
          data: audioBase64,
        },
      });
    } else {
      console.log(
        `[POST /api/ask-agronomist] textQuestion="${question.trim()}", activeLanguage=${safeLang}`
      );
    }

    const systemInstruction = [
      'You are a friendly South Indian agronomist and crop doctor for UzhavanVoice.',
      `Answer the farmer's question in simple, practical, farmer-friendly ${languageLabel(safeLang)}.`,
      'Keep the answer to 3 clear spoken sentences with actionable organic and safe chemical guidance.',
    ].join(' ');

    const { response, model: usedModel } = await generateWithFallback({
      contents: [{ role: 'user', parts }],
      config: {
        systemInstruction,
        responseMimeType: 'application/json',
        responseSchema: AGRONOMIST_QA_SCHEMA,
        temperature: 0.25,
      },
    });

    const rawText = response.text || '{}';
    console.log(`[POST /api/ask-agronomist] (${usedModel}) Gemini raw JSON:`, rawText);

    const parsed = JSON.parse(rawText);

    return res.json({
      answer: parsed.answer || '',
      topic: parsed.topic || 'Crop Advisory',
      cropName: parsed.cropName || null,
      languageCode: safeLang,
    });
  } catch (err) {
    console.error('Error in /api/ask-agronomist:', err);
    return res.status(500).json({
      error: err instanceof Error ? err.message : 'Agronomist Q&A failed',
    });
  }
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(
    `F-Vision backend listening on http://0.0.0.0:${PORT} (primary model: ${PRIMARY_MODEL})`
  );
});
