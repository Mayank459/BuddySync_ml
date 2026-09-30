# Demo environment for BuddySync's trip planner. Copy to demo_env.ps1 (never commit that file), fill in keys,
# then in every PowerShell window you use for the demo:   . .\demo_env.ps1
# Tip: on demo morning create a NEW Gemini key in a NEW Google Cloud project: the free tier is 20 requests/day/model.

# Chat and drafts: Groq first (fast), Gemini as the fallback
$env:LLM_BASE_URL = "https://api.groq.com/openai/v1"
$env:LLM_API_KEY = "<groq key: gsk_...>"
$env:LLM_MODEL = "qwen/qwen3.8-27b,openai/gpt-oss-120b,openai/gpt-oss-20b"  # each has its own free quota
$env:LLM_FALLBACK_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
$env:LLM_FALLBACK_API_KEY = "<gemini key>"
$env:LLM_FALLBACK_MODEL = "gemini-flash-lite-latest,gemini-3.8-flash"

# Live prices, affiliate links, voice
$env:LITEAPI_KEY = "<sand_... or prod_...>"
$env:TRAVELPAYOUTS_TOKEN = "<travelpayouts token>"
$env:TRAVELPAYOUTS_MARKER = "783614"
$env:STT_API_KEY = "<groq key: gsk_...>"
$env:API_URL = "http://127.0.0.1:8765"          # where the playground finds the API
