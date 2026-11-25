import dns from 'node:dns';
dns.setDefaultResultOrder('ipv4first');

import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import dotenv from 'dotenv';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Load .env from backend directory first, then fallback to current working directory
dotenv.config({ path: join(__dirname, '.env') });
dotenv.config();

import express from 'express';
import cors from 'cors';
import rateLimit from 'express-rate-limit';
import { GoogleGenAI, Type } from '@google/genai';

const app = express();
const PORT = Number(process.env.PORT) || 3000;

// Model fallback cascade: primary fast model -> secondary -> tertiary
export const HARDCODED_MODELS = [
  'gemini-3.5-flash-lite',
  'gemini-3-flash',
  'gemini-2.5-flash',
];
export const PRIMARY_MODEL = HARDCODED_MODELS[0];
export const FALLBACK_MODEL = HARDCODED_MODELS[1];
const REQUEST_TIMEOUT_MS = Number(process.env.GEMINI_TIMEOUT_MS) || 15000;
const THINKING_LEVEL = process.env.GEMINI_THINKING_LEVEL || 'low';

// Middleware
app.use(cors());

// Rate limit /api/* routes (~20 req/min per IP)
const apiLimiter =
  process.env.NODE_ENV === 'test'
    ? (_req, _res, next) => next()
    : rateLimit({
        windowMs: 60 * 1000,
        max: 20,
        standardHeaders: true,
        legacyHeaders: false,
        message: { error: 'Too many requests, please slow down and try again later.' },
      });
app.use('/api', apiLimiter);

// 6 MB payload limit with clear 413 handling
app.use(express.json({ limit: '6mb' }));
app.use((err, _req, res, next) => {
  if (err?.type === 'entity.too.large' || err?.status === 413) {
    return res.status(413).json({
      error: 'Payload too large. Maximum payload size is 6 MB.',
    });
  }
  next(err);
});

// Shared secret authorization middleware for all /api/* routes
app.use('/api', (req, res, next) => {
  const configuredKey = process.env.APP_KEY?.trim();
  if (!configuredKey) {
    return next();
  }
  const clientKey = req.headers['x-app-key'];
  if (clientKey !== configuredKey) {
    return res.status(401).json({
      error: 'Unauthorized: Missing or invalid x-app-key header',
    });
  }
  next();
});

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

export function isRetryableFallbackError(err) {
  if (!err) return false;

  const status =
    err.status ??
    err.statusCode ??
    err.code ??
    err?.error?.code ??
    err?.error?.status ??
    err?.response?.status;

  // Catch 404 NOT_FOUND, 429 RESOURCE_EXHAUSTED, 503 UNAVAILABLE, and timeouts
  if (
    status === 404 ||
    status === '404' ||
    status === 'NOT_FOUND' ||
    status === 429 ||
    status === '429' ||
    status === 'RESOURCE_EXHAUSTED' ||
    status === 503 ||
    status === '503' ||
    status === 'UNAVAILABLE' ||
    status === 'ETIMEDOUT' ||
    status === 'ABORT_ERR' ||
    err.name === 'TimeoutError' ||
    err.name === 'NotFoundError' ||
    err.name === 'RateLimitError'
  ) {
    return true;
  }

  const msg = String(
    err.message ||
    err?.error?.message ||
    err?.body ||
    err
  ).toLowerCase();

  return (
    // 404 / Retired models / Not Found
    msg.includes('404') ||
    msg.includes('not_found') ||
    msg.includes('not found') ||
    msg.includes('no longer available') ||
    msg.includes('is not supported') ||
    msg.includes('is not available') ||
    msg.includes('does not exist') ||
    msg.includes('call listmodels') ||
    // 429 / Quota / Rate limits
    msg.includes('429') ||
    msg.includes('resource_exhausted') ||
    msg.includes('quota') ||
    msg.includes('rate limit') ||
    msg.includes('too many requests') ||
    // 503 / Unavailable / Overloaded
    msg.includes('503') ||
    msg.includes('unavailable') ||
    msg.includes('overloaded') ||
    msg.includes('high demand') ||
    msg.includes('service unavailable') ||
    // Timeouts and aborts
    msg.includes('timeout') ||
    msg.includes('timed out') ||
    msg.includes('aborted') ||
    msg.includes('abort_err') ||
    msg.includes('econnreset') ||
    msg.includes('etimedout')
  );
}

export const isRetryableOverloadOrTimeout = isRetryableFallbackError;

const DEPRECATED_MODEL_PREFIXES = ['gemini-2.0', 'gemini-1.5', 'gemini-1.0'];
const EXCLUDED_CAPABILITY_SUBSTRINGS = ['embedding', 'imagen', 'veo', 'tts', 'image'];

/**
 * Automatically queries the Gemini API for available models that support generateContent,
 * filtering for gemini or gemma models and preferring flash-lite or flash.
 */
export async function discoverDynamicModels(excludeModels = []) {
  try {
    let apiKey = process.env.GEMINI_API_KEY?.trim();
    if (!apiKey) {
      try {
        apiKey = getGeminiApiKey();
      } catch (_) {}
    }
    if (!apiKey) {
      console.warn('[DynamicDiscovery] GEMINI_API_KEY is not available');
      return [];
    }

    const url = `https://generativelanguage.googleapis.com/v1beta/models?key=${encodeURIComponent(apiKey)}`;
    const res = await fetch(url);
    if (!res.ok) {
      console.warn(`[DynamicDiscovery] Failed to fetch models (${res.status}): ${await res.text()}`);
      return [];
    }

    const data = await res.json();
    const models = Array.isArray(data.models) ? data.models : [];
    const excludeSet = new Set(
      (Array.isArray(excludeModels) ? excludeModels : [excludeModels])
        .filter(Boolean)
        .map((m) => String(m).toLowerCase().replace(/^models\//, ''))
    );

    const candidates = models
      .filter((m) => {
        const methods = Array.isArray(m.supportedGenerationMethods)
          ? m.supportedGenerationMethods
          : [];
        if (!methods.includes('generateContent')) return false;

        const id = String(m.name || '').replace(/^models\//, '').toLowerCase();
        if (!id || excludeSet.has(id)) return false;

        // Must be a gemini or gemma model
        if (!id.startsWith('gemini') && !id.startsWith('gemma')) {
          return false;
        }

        // Exclude legacy deprecated model versions
        if (DEPRECATED_MODEL_PREFIXES.some((prefix) => id.startsWith(prefix))) {
          return false;
        }

        // Exclude non-text/specialized capability models (embeddings, tts, imagen, veo)
        if (EXCLUDED_CAPABILITY_SUBSTRINGS.some((ex) => id.includes(ex))) {
          return false;
        }

        return true;
      })
      .map((m) => String(m.name).replace(/^models\//, ''))
      .sort((a, b) => {
        const aId = a.toLowerCase();
        const bId = b.toLowerCase();

        // Preference order: flash-lite (0) > flash (1) > gemini (2) > gemma (3) > other (4)
        const getScore = (id) => {
          if (id.includes('flash-lite')) return 0;
          if (id.includes('flash')) return 1;
          if (id.startsWith('gemini')) return 2;
          if (id.startsWith('gemma')) return 3;
          return 4;
        };

        const scoreA = getScore(aId);
        const scoreB = getScore(bId);
        if (scoreA !== scoreB) {
          return scoreA - scoreB;
        }

        // Tie-breaker: sort descending to prefer newer versions (e.g. 3.5 > 3.0 > 2.5)
        return bId.localeCompare(aId);
      });

    console.log(`[DynamicDiscovery] Discovered ${candidates.length} candidate model(s):`, candidates);
    return candidates;
  } catch (err) {
    console.warn('[DynamicDiscovery] Error listing models from Gemini API:', err?.message || err);
    return [];
  }
}

export const listAvailableFallbackModels = discoverDynamicModels;

async function callWithTimeout(ai, params, model) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => {
    controller.abort();
  }, REQUEST_TIMEOUT_MS);

  const startMs = Date.now();
  try {
    const config = { ...(params.config || {}) };
    const modelLower = model.toLowerCase();

    // thinkingLevel is Gemini 3+ specific; remove for Gemini 2.x and Gemma to prevent INVALID_ARGUMENT
    if (modelLower.startsWith('gemini-2') || modelLower.startsWith('gemma')) {
      delete config.thinkingConfig;
    }

    // gemma models do not support responseSchema / responseMimeType in API config
    if (modelLower.startsWith('gemma')) {
      delete config.responseSchema;
      delete config.responseMimeType;
    }

    const response = await ai.models.generateContent({
      ...params,
      model,
      config: {
        ...config,
        abortSignal: controller.signal,
      },
    });
    const duration = Date.now() - startMs;
    return { response, duration };
  } catch (err) {
    const duration = Date.now() - startMs;
    if (controller.signal.aborted) {
      const timeoutErr = new Error(`Gemini request timed out after ${REQUEST_TIMEOUT_MS}ms`);
      timeoutErr.name = 'TimeoutError';
      timeoutErr.duration = duration;
      throw timeoutErr;
    }
    err.duration = duration;
    throw err;
  } finally {
    clearTimeout(timeoutId);
  }
}

function parseModelJsonResponse(rawText) {
  if (!rawText || typeof rawText !== 'string') return {};
  const cleaned = rawText.trim();
  try {
    return JSON.parse(cleaned);
  } catch (_) {}

  const codeBlockMatch = cleaned.match(/```(?:json)?\s*([\s\S]*?)\s*```/i);
  if (codeBlockMatch) {
    try {
      return JSON.parse(codeBlockMatch[1].trim());
    } catch (_) {}
  }

  const firstBrace = cleaned.indexOf('{');
  const lastBrace = cleaned.lastIndexOf('}');
  if (firstBrace !== -1 && lastBrace > firstBrace) {
    try {
      return JSON.parse(cleaned.slice(firstBrace, lastBrace + 1));
    } catch (_) {}
  }

  throw new Error(`Failed to parse JSON from AI response: ${cleaned.slice(0, 100)}`);
}

/**
 * Executes a Gemini request with non-repeating fallbacks:
 * 1. Hardcoded model cascade:
 *    - gemini-3.5-flash-lite (primary fast model)
 *    - gemini-3-flash
 *    - gemini-2.5-flash
 * 2. Automatic dynamic model discovery fallback:
 *    If ALL hardcoded models fail with 404 NOT_FOUND, 429 RESOURCE_EXHAUSTED,
 *    or 503 UNAVAILABLE, automatically fetches active models via
 *    GET https://generativelanguage.googleapis.com/v1beta/models?key=${process.env.GEMINI_API_KEY},
 *    picks the first available gemini or gemma model supporting generateContent
 *    (preferring flash-lite or flash), and retries the request immediately!
 */
export async function generateWithFallback(params, tag = 'Gemini') {
  const ai = getGeminiClient();
  const t0 = Date.now();

  const modelsToTry = [...HARDCODED_MODELS];
  let dynamicModelsFetched = false;
  let lastErr = null;

  for (let i = 0; i < modelsToTry.length; i++) {
    const currentModel = modelsToTry[i];
    const attempt = i + 1;
    try {
      const { response, duration } = await callWithTimeout(ai, params, currentModel);
      console.log(`[${tag}] model=${currentModel} attempt=${attempt} duration=${duration}ms status=success`);
      return { response, model: currentModel, totalDuration: Date.now() - t0 };
    } catch (err) {
      lastErr = err;
      const duration = err.duration || 0;
      console.warn(`[${tag}] ${currentModel} attempt ${attempt} failed in ${duration}ms (${err?.message || err})`);

      if (!isRetryableFallbackError(err)) {
        console.error(`[${tag}] Non-retryable error on model ${currentModel}:`, err);
        throw err;
      }

      // If ALL hardcoded models have failed with 404, 429, or 503,
      // trigger dynamic model discovery fallback immediately!
      if (i === modelsToTry.length - 1 && !dynamicModelsFetched) {
        dynamicModelsFetched = true;
        console.warn(
          `[${tag}] All hardcoded models (${HARDCODED_MODELS.join(', ')}) failed with retryable status (${err?.status || err?.code || err?.message || err}). Fetching dynamic models from Gemini API...`
        );
        try {
          const dynamicModels = await discoverDynamicModels(modelsToTry);
          if (dynamicModels.length > 0) {
            console.log(
              `[${tag}] Dynamically picked fallback model "${dynamicModels[0]}" (found ${dynamicModels.length} candidate(s)). Retrying immediately...`
            );
            for (const dm of dynamicModels) {
              if (!modelsToTry.includes(dm)) {
                modelsToTry.push(dm);
              }
            }
          } else {
            console.warn(`[${tag}] Dynamic discovery returned 0 candidates.`);
          }
        } catch (dynErr) {
          console.warn(`[${tag}] Could not list dynamic fallback models:`, dynErr?.message || dynErr);
        }
      }
    }
  }

  throw lastErr || new Error(`[${tag}] All fallback models failed.`);
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
        'Identified disease, pest, nutrient deficiency, or "Healthy Crop" if no issue.',
    },
    isHealthy: {
      type: Type.BOOLEAN,
      description:
        'True if the crop is healthy with no pests/diseases; false if infected/deficient.',
    },
    severity: {
      type: Type.STRING,
      enum: ['Low', 'Moderate', 'High'],
      description: 'Severity level of the issue.',
    },
    voiceSummary: {
      type: Type.STRING,
      description:
        'A single concise 1-sentence farmer-friendly spoken summary (under 20 words).',
    },
    treatmentSteps: {
      type: Type.ARRAY,
      maxItems: 4,
      description: 'Ordered day-by-day practical field treatment steps (at most 4 short steps).',
      items: {
        type: Type.OBJECT,
        properties: {
          dayNumber: {
            type: Type.INTEGER,
            description: 'Day number (e.g. 1, 2, 4).',
          },
          title: {
            type: Type.STRING,
            description: 'Short action title in activeLanguage.',
          },
          instruction: {
            type: Type.STRING,
            description:
              'Short 1-sentence practical field instruction in activeLanguage.',
          },
        },
        required: ['dayNumber', 'title', 'instruction'],
      },
    },
  },
  required: [
    'cropName',
    'diseaseName',
    'isHealthy',
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

// Instant health endpoint (public, zero Gemini calls)
app.get('/health', (_req, res) => {
  res.json({
    ok: true,
    primaryModel: PRIMARY_MODEL,
    fallbackModel: FALLBACK_MODEL,
    hardcodedModels: HARDCODED_MODELS,
    hasApiKey: Boolean(process.env.GEMINI_API_KEY?.trim()),
  });
});

function matchVoiceFastPath(text, activeLanguage = 'en-IN') {
  if (!text || typeof text !== 'string') return null;
  const raw = text.trim();
  if (!raw) return null;
  const lower = raw.toLowerCase();

  const lang = VALID_LANGUAGES.includes(activeLanguage) ? activeLanguage : 'en-IN';

  // Language switching (Tamil, Telugu, English)
  if (lower.includes('tamil') || lower.includes('தமிழ்') || lower.includes('తమిళ')) {
    return {
      intent: 'CHANGE_LANGUAGE',
      targetLanguage: 'ta-IN',
      targetCrop: null,
      acres: null,
      spokenReply: 'மொழியைத் தமிழாக மாற்றியுள்ளேன்.',
      fastPath: true,
    };
  }
  if (lower.includes('telugu') || lower.includes('தெலுங்கு') || lower.includes('తెలుగు')) {
    return {
      intent: 'CHANGE_LANGUAGE',
      targetLanguage: 'te-IN',
      targetCrop: null,
      acres: null,
      spokenReply: 'భాషను తెలుగుగా మార్చాను.',
      fastPath: true,
    };
  }
  if (
    lower.includes('english') ||
    lower.includes('ஆங்கிலம்') ||
    lower.includes('ఇంగ్లీష్') ||
    lower.includes('ఇంగ్లీషు')
  ) {
    return {
      intent: 'CHANGE_LANGUAGE',
      targetLanguage: 'en-IN',
      targetCrop: null,
      acres: null,
      spokenReply: 'Language switched to English.',
      fastPath: true,
    };
  }

  // Next / Previous / Back navigation
  if (
    lower.includes('next') ||
    lower.includes('அடுத்து') ||
    lower.includes('அடுத்த') ||
    lower.includes('తర్వాత') ||
    lower.includes('తర్వాతి') ||
    lower.includes('తదుపరి')
  ) {
    const replies = {
      'ta-IN': 'அடுத்த படிக்குச் செல்கிறேன்.',
      'te-IN': 'తదుపరి దశకు వెళ్తున్నాను.',
      'en-IN': 'Moving to the next step.',
    };
    return {
      intent: 'NEXT_STEP',
      targetLanguage: null,
      targetCrop: null,
      acres: null,
      spokenReply: replies[lang] || replies['en-IN'],
      fastPath: true,
    };
  }

  if (
    lower.includes('back') ||
    lower.includes('previous') ||
    lower.includes('பின்னால்') ||
    lower.includes('முந்தைய') ||
    lower.includes('பின்') ||
    lower.includes('వెనక్కి') ||
    lower.includes('వెనుకకు') ||
    lower.includes('మునుపటి')
  ) {
    const replies = {
      'ta-IN': 'முந்தைய திரைக்குச் செல்கிறேன்.',
      'te-IN': 'మునుపటి స్క్రీన్‌కు వెళ్తున్నాను.',
      'en-IN': 'Going back.',
    };
    return {
      intent: 'GO_BACK',
      targetLanguage: null,
      targetCrop: null,
      acres: null,
      spokenReply: replies[lang] || replies['en-IN'],
      fastPath: true,
    };
  }

  // Scanner
  if (
    lower.includes('scanner') ||
    lower.includes('scan') ||
    lower.includes('camera') ||
    lower.includes('leaf') ||
    lower.includes('ஸ்கேனர்') ||
    lower.includes('ஸ்கேன்') ||
    lower.includes('స్కానర్') ||
    lower.includes('స్కాన్')
  ) {
    const replies = {
      'ta-IN': 'பயிர் ஸ்கேனரைத் திறக்கிறேன்.',
      'te-IN': 'పంట స్కానర్‌ను తెరుస్తున్నాను.',
      'en-IN': 'Opening crop scanner.',
    };
    return {
      intent: 'OPEN_SCANNER',
      targetLanguage: null,
      targetCrop: null,
      acres: null,
      spokenReply: replies[lang] || replies['en-IN'],
      fastPath: true,
    };
  }

  // Weather
  if (
    lower.includes('weather') ||
    lower.includes('rain') ||
    lower.includes('climate') ||
    lower.includes('வானிலை') ||
    lower.includes('மழை') ||
    lower.includes('వాతావరణ') ||
    lower.includes('వర్షం')
  ) {
    const replies = {
      'ta-IN': 'வானிலை விவரங்களைத் திறக்கிறேன்.',
      'te-IN': 'వాతావరణ వివరాలను తెరుస్తున్నాను.',
      'en-IN': 'Opening weather forecast.',
    };
    return {
      intent: 'OPEN_WEATHER',
      targetLanguage: null,
      targetCrop: null,
      acres: null,
      spokenReply: replies[lang] || replies['en-IN'],
      fastPath: true,
    };
  }

  // Fertilizer
  if (
    lower.includes('fertilizer') ||
    lower.includes('urea') ||
    lower.includes('உரம்') ||
    lower.includes('எரு') ||
    lower.includes('ఎరువు') ||
    lower.includes('యూరియా')
  ) {
    let targetCrop = null;
    if (lower.includes('paddy') || lower.includes('rice') || lower.includes('நெல்') || lower.includes('వరి')) targetCrop = 'paddy';
    else if (lower.includes('tomato') || lower.includes('தக்காளி') || lower.includes('టమోటా')) targetCrop = 'tomato';
    else if (lower.includes('banana') || lower.includes('வாழை') || lower.includes('అరటి')) targetCrop = 'banana';
    else if (lower.includes('cotton') || lower.includes('பருத்தி') || lower.includes('పత్తి')) targetCrop = 'cotton';
    else if (lower.includes('groundnut') || lower.includes('நிலக்கடலை') || lower.includes('వేరుశనగ')) targetCrop = 'groundnut';
    else if (lower.includes('chilli') || lower.includes('மிளகாய்') || lower.includes('మిరప')) targetCrop = 'chilli';

    const acresMatch = lower.match(/(\d+(?:\.\d+)?)\s*(?:acres?|acre|ஏக்கர்|ఎకరాలు?)/i);
    const acres = acresMatch ? parseFloat(acresMatch[1]) : null;

    const replies = {
      'ta-IN': 'உரக் கணக்கீட்டுப் பக்கத்தைத் திறக்கிறேன்.',
      'te-IN': 'ఎరువుల కాలిక్యులేటర్‌ను తెరుస్తున్నాను.',
      'en-IN': 'Opening fertilizer calculator.',
    };
    return {
      intent: 'OPEN_FERTILIZER',
      targetLanguage: null,
      targetCrop,
      acres,
      spokenReply: replies[lang] || replies['en-IN'],
      fastPath: true,
    };
  }

  // Calendar
  if (
    lower.includes('calendar') ||
    lower.includes('schedule') ||
    lower.includes('cultivation') ||
    lower.includes('நாட்காட்டி') ||
    lower.includes('காலண்டர்') ||
    lower.includes('క్యాలెండర్') ||
    lower.includes('కాలెండర్')
  ) {
    let targetCrop = null;
    if (lower.includes('paddy') || lower.includes('rice') || lower.includes('நெல்') || lower.includes('వరి')) targetCrop = 'paddy';
    else if (lower.includes('tomato') || lower.includes('தக்காளி') || lower.includes('టమోటా')) targetCrop = 'tomato';
    else if (lower.includes('banana') || lower.includes('வாழை') || lower.includes('అరటి')) targetCrop = 'banana';
    else if (lower.includes('cotton') || lower.includes('பருத்தி') || lower.includes('పత్తి')) targetCrop = 'cotton';
    else if (lower.includes('groundnut') || lower.includes('நிலக்கடலை') || lower.includes('వేరుశనగ')) targetCrop = 'groundnut';
    else if (lower.includes('chilli') || lower.includes('மிளகாய்') || lower.includes('మిరప')) targetCrop = 'chilli';

    const replies = {
      'ta-IN': 'சாகுபடி நாட்காட்டியைத் திறக்கிறேன்.',
      'te-IN': 'సాగు క్యాలెండర్‌ను తెరుస్తున్నాను.',
      'en-IN': 'Opening crop cultivation calendar.',
    };
    return {
      intent: 'OPEN_CALENDAR',
      targetLanguage: null,
      targetCrop,
      acres: null,
      spokenReply: replies[lang] || replies['en-IN'],
      fastPath: true,
    };
  }

  // Pest Library
  if (
    lower.includes('pest') ||
    lower.includes('insect') ||
    lower.includes('disease') ||
    lower.includes('பூச்சி') ||
    lower.includes('புరుగు') ||
    lower.includes('చీడ')
  ) {
    const replies = {
      'ta-IN': 'பூச்சி மற்றும் நோய் நூலகத்தைத் திறக்கிறேன்.',
      'te-IN': 'చీడపీడల లైబ్రరీని తెరుస్తున్నాను.',
      'en-IN': 'Opening pest and disease library.',
    };
    return {
      intent: 'OPEN_PEST_LIBRARY',
      targetLanguage: null,
      targetCrop: null,
      acres: null,
      spokenReply: replies[lang] || replies['en-IN'],
      fastPath: true,
    };
  }

  // Soil Sensor / Moisture
  if (
    lower.includes('moisture') ||
    lower.includes('erapatham') ||
    lower.includes('ஈரப்பதம்') ||
    lower.includes('తేమ') ||
    (lower.includes('soil') && (lower.includes('check') || lower.includes('level') || lower.includes('evlo') || lower.includes('cheppu') || lower.includes('sollu')))
  ) {
    const replies = {
      'ta-IN': 'மண் ஈரப்பத அளவைச் சரிபார்க்கிறேன்.',
      'te-IN': 'నేల తేమ స్థాయిని తనిఖీ చేస్తున్నాను.',
      'en-IN': 'Checking soil moisture level.',
    };
    return {
      intent: 'CHECK_SOIL_MOISTURE',
      targetLanguage: null,
      targetCrop: null,
      acres: null,
      spokenReply: replies[lang] || replies['en-IN'],
      fastPath: true,
    };
  }

  if (
    lower.includes('soil') ||
    lower.includes('sensor') ||
    lower.includes('iot') ||
    lower.includes('மண்') ||
    lower.includes('சென்சார்') ||
    lower.includes('నేల') ||
    lower.includes('సెన్సార్')
  ) {
    const replies = {
      'ta-IN': 'மண் சென்சார் திரையைத் திறக்கிறேன்.',
      'te-IN': 'నేల సెన్సార్ స్క్రీన్‌ను తెరుస్తున్నాను.',
      'en-IN': 'Opening soil sensor screen.',
    };
    return {
      intent: 'OPEN_SOIL_SENSOR',
      targetLanguage: null,
      targetCrop: null,
      acres: null,
      spokenReply: replies[lang] || replies['en-IN'],
      fastPath: true,
    };
  }

  // Home
  if (
    lower.includes('home') ||
    lower.includes('main') ||
    lower.includes('முகப்பு') ||
    lower.includes('ஹோம்') ||
    lower.includes('హోమ్') ||
    lower.includes('హోం')
  ) {
    const replies = {
      'ta-IN': 'முகப்புப் பக்கத்திற்குச் செல்கிறேன்.',
      'te-IN': 'హోమ్ స్క్రీన్‌కు వెళ్తున్నాను.',
      'en-IN': 'Navigating to home.',
    };
    return {
      intent: 'OPEN_HOME',
      targetLanguage: null,
      targetCrop: null,
      acres: null,
      spokenReply: replies[lang] || replies['en-IN'],
      fastPath: true,
    };
  }

  // Menu
  if (
    lower.includes('menu') ||
    lower.includes('மெனு') ||
    lower.includes('பட்டியல்') ||
    lower.includes('మెనూ') ||
    lower.includes('మెను')
  ) {
    const replies = {
      'ta-IN': 'மெனுவைத் திறக்கிறேன்.',
      'te-IN': 'మెనూను తెరుస్తున్నాను.',
      'en-IN': 'Opening menu.',
    };
    return {
      intent: 'OPEN_MENU',
      targetLanguage: null,
      targetCrop: null,
      acres: null,
      spokenReply: replies[lang] || replies['en-IN'],
      fastPath: true,
    };
  }

  return null;
}

app.post('/api/voice-intent', async (req, res) => {
  const reqStart = Date.now();
  try {
    const {
      audioBase64,
      mimeType = 'audio/m4a',
      activeLanguage = 'en-IN',
      currentScreen = 'home',
      transcript,
      text,
      command,
      query,
    } = req.body ?? {};

    const safeLang = VALID_LANGUAGES.includes(activeLanguage)
      ? activeLanguage
      : 'en-IN';

    // Zero-Quota Fast-Path: check text/transcript against basic navigation keywords
    const rawTranscript =
      (typeof transcript === 'string' && transcript.trim()) ||
      (typeof text === 'string' && text.trim()) ||
      (typeof command === 'string' && command.trim()) ||
      (typeof query === 'string' && query.trim()) ||
      '';

    if (rawTranscript) {
      const fastResult = matchVoiceFastPath(rawTranscript, safeLang);
      if (fastResult) {
        console.log(
          `[POST /api/voice-intent] FAST-PATH hit for "${rawTranscript}": intent=${fastResult.intent} (0 Gemini quota spent)`
        );
        return res.json({
          intent: fastResult.intent,
          targetLanguage: fastResult.targetLanguage,
          targetCrop: fastResult.targetCrop,
          acres: fastResult.acres,
          spokenReply: fastResult.spokenReply,
          fastPath: true,
        });
      }
    }

    if (!audioBase64 || typeof audioBase64 !== 'string') {
      if (!rawTranscript) {
        return res.status(400).json({ error: 'audioBase64 or transcript/text is required' });
      }
    }

    const parts = [
      {
        text: rawTranscript
          ? `Classify the voice command or farming question: "${rawTranscript}" and return structured JSON.`
          : 'Classify the voice command or farming question in the attached audio and return structured JSON.',
      },
    ];

    if (audioBase64 && typeof audioBase64 === 'string') {
      const audioBytes = Buffer.from(audioBase64, 'base64').length;
      const rawMime = String(mimeType).toLowerCase().trim();
      const geminiAudioMime = SUPPORTED_AUDIO_MIME_MAP[rawMime] || 'audio/mp4';
      parts.push({
        inlineData: {
          mimeType: geminiAudioMime,
          data: audioBase64,
        },
      });
      console.log(
        `[POST /api/voice-intent] audioBytes=${audioBytes}, mimeType=${rawMime}, activeLanguage=${safeLang}, currentScreen=${currentScreen}`
      );
    } else {
      console.log(
        `[POST /api/voice-intent] textTranscript="${rawTranscript}", activeLanguage=${safeLang}, currentScreen=${currentScreen}`
      );
    }

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

    const { response, model: usedModel, totalDuration } = await generateWithFallback(
      {
        contents: [
          {
            role: 'user',
            parts,
          },
        ],
        config: {
          systemInstruction,
          responseMimeType: 'application/json',
          responseSchema: VOICE_INTENT_RESPONSE_SCHEMA,
          thinkingConfig: {
            thinkingLevel: THINKING_LEVEL,
          },
        },
      },
      'VOICE_INTENT'
    );

    const rawText = response.text || '{}';
    let parsed;
    try {
      parsed = parseModelJsonResponse(rawText);
    } catch {
      console.error(`[POST /api/voice-intent] (${usedModel}) JSON parse error:`, rawText);
      return res.status(502).json({ error: 'AI model returned an unparseable response' });
    }

    const intent = VALID_INTENTS.includes(parsed.intent)
      ? parsed.intent
      : 'UNKNOWN';
    const targetLanguage = VALID_LANGUAGES.includes(parsed.targetLanguage)
      ? parsed.targetLanguage
      : null;
    const targetCrop = VALID_CROPS.includes(parsed.targetCrop)
      ? parsed.targetCrop
      : null;
    const acres =
      typeof parsed.acres === 'number' && Number.isFinite(parsed.acres) && parsed.acres > 0
        ? parsed.acres
        : null;

    console.log(
      `[POST /api/voice-intent] model=${usedModel} totalDuration=${totalDuration}ms overallMs=${Date.now() - reqStart} intent=${intent}`
    );

    return res.json({
      intent,
      targetLanguage,
      targetCrop,
      acres,
      spokenReply:
        typeof parsed.spokenReply === 'string' ? parsed.spokenReply : '',
      fastPath: false,
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
  const reqStart = Date.now();
  try {
    const {
      imageBase64,
      mimeType = 'image/jpeg',
      activeLanguage = 'en-IN',
    } = req.body ?? {};

    if (!imageBase64 || typeof imageBase64 !== 'string') {
      return res.status(400).json({ error: 'imageBase64 is required' });
    }

    const rawMime = String(mimeType).toLowerCase().trim();
    const validMimes = ['image/jpeg', 'image/png', 'image/webp'];
    if (!validMimes.includes(rawMime)) {
      return res.status(400).json({
        error: 'Invalid mimeType. Allowed: image/jpeg, image/png, image/webp',
      });
    }

    const imageBytes = Buffer.from(imageBase64, 'base64').length;
    if (imageBytes > 5 * 1024 * 1024) {
      return res.status(413).json({ error: 'Image exceeds 5MB size limit' });
    }

    const safeLang = VALID_LANGUAGES.includes(activeLanguage)
      ? activeLanguage
      : 'en-IN';

    console.log(
      `[POST /api/diagnose-leaf] bytes=${imageBytes}, mimeType=${rawMime}, activeLanguage=${safeLang}`
    );

    const systemInstruction = [
      'You are an ultra-fast expert crop doctor for South Indian farmers.',
      'Analyze the crop leaf image to identify the crop and whether it is healthy or diseased.',
      'If healthy: set isHealthy to true, diseaseName to "Healthy Crop", severity to "Low", voiceSummary to 1 punchy congratulatory sentence (under 15 words), and 1-2 brief maintenance tips in treatmentSteps.',
      'If diseased/deficient: set isHealthy to false, accurately identify cropName, diseaseName, severity, 1 concise spoken sentence in voiceSummary (under 20 words), and at most 4 short actionable bullet treatmentSteps with ONE short sentence each.',
      `Write all output text strictly in simple, farmer-friendly ${languageLabel(safeLang)}.`,
      'Do NOT write long paragraphs. Keep instructions to ONE short sentence per step for ultra-fast generation.',
    ].join(' ');

    const { response, model: usedModel, totalDuration } = await generateWithFallback(
      {
        contents: [
          {
            role: 'user',
            parts: [
              {
                text: `Diagnose this crop leaf image and return structured JSON in ${languageLabel(safeLang)}.`,
              },
              {
                inlineData: {
                  mimeType: rawMime,
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
          maxOutputTokens: 600,
          thinkingConfig: {
            thinkingLevel: THINKING_LEVEL,
          },
        },
      },
      'DIAGNOSE_LEAF'
    );

    const rawText = response.text || '{}';
    let parsed;
    try {
      parsed = parseModelJsonResponse(rawText);
    } catch {
      console.error(`[POST /api/diagnose-leaf] (${usedModel}) JSON parse error:`, rawText);
      return res.status(502).json({ error: 'AI model returned an unparseable response' });
    }

    const diseaseName = parsed.diseaseName || 'Healthy Crop';
    const isHealthy = typeof parsed.isHealthy === 'boolean'
      ? parsed.isHealthy
      : /healthy|no disease|ஆரோக்கிய|ఆరోగ్య/i.test(diseaseName);

    console.log(
      `[POST /api/diagnose-leaf] model=${usedModel} totalDuration=${totalDuration}ms overallMs=${Date.now() - reqStart} crop=${parsed.cropName || 'Unknown'}`
    );

    return res.json({
      cropName: parsed.cropName || 'Unknown Crop',
      diseaseName,
      isHealthy,
      severity: ['Low', 'Moderate', 'High'].includes(parsed.severity)
        ? parsed.severity
        : 'Moderate',
      voiceSummary: parsed.voiceSummary || '',
      languageCode: safeLang,
      treatmentSteps: Array.isArray(parsed.treatmentSteps)
        ? parsed.treatmentSteps.slice(0, 4)
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
  const reqStart = Date.now();
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
        `[POST /api/ask-agronomist] audioBytes=${audioBytes}, mimeType=${rawMime}, activeLanguage=${safeLang}`
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

    const { response, model: usedModel, totalDuration } = await generateWithFallback(
      {
        contents: [{ role: 'user', parts }],
        config: {
          systemInstruction,
          responseMimeType: 'application/json',
          responseSchema: AGRONOMIST_QA_SCHEMA,
          thinkingConfig: {
            thinkingLevel: THINKING_LEVEL,
          },
        },
      },
      'ASK_AGRONOMIST'
    );

    const rawText = response.text || '{}';
    let parsed;
    try {
      parsed = parseModelJsonResponse(rawText);
    } catch {
      console.error(`[POST /api/ask-agronomist] (${usedModel}) JSON parse error:`, rawText);
      return res.status(502).json({ error: 'AI model returned an unparseable response' });
    }

    console.log(
      `[POST /api/ask-agronomist] model=${usedModel} totalDuration=${totalDuration}ms overallMs=${Date.now() - reqStart}`
    );

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

// Process crash prevention
process.on('unhandledRejection', (reason, promise) => {
  console.error('[Process] Unhandled Rejection at:', promise, 'reason:', reason);
});

process.on('uncaughtException', (err) => {
  console.error('[Process] Uncaught Exception:', err);
});

if (process.env.NODE_ENV !== 'test') {
  app.listen(PORT, '0.0.0.0', () => {
    console.log(
      `F-Vision backend listening on http://0.0.0.0:${PORT} (models: ${HARDCODED_MODELS.join(' -> ')}, thinkingLevel: ${THINKING_LEVEL})`
    );
  });
}
