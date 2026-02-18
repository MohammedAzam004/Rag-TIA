# ElevenLabs API Key Troubleshooting Guide

## Problem: "Error: Unauthorized. Check your ElevenLabs API key"

This error occurs when ElevenLabs rejects your API key. Here are the solutions:

---

## Quick Diagnosis

Run the test script to check your API key:
```bash
python test_elevenlabs.py
```

---

## Common Causes & Solutions

### 1. **Invalid or Expired API Key**
**Symptoms:** Immediate 401 Unauthorized error

**Solution:**
1. Go to https://elevenlabs.io/app/settings/api-keys
2. Delete old API key
3. Click "Create New Key"
4. Copy the new key (starts with `sk_`)
5. Paste it in the Streamlit app
6. Click the "🔍 Test Key" button to verify

### 2. **Free Tier Quota Exceeded**
**Symptoms:** Worked before, now getting 401 or quota errors

**Solution:**
- Check your quota at: https://elevenlabs.io/app/usage
- Wait for monthly reset, OR
- Upgrade to paid plan, OR
- Create a new account with different email

### 3. **Account Suspended**
**Symptoms:** All API keys fail, even new ones

**Solution:**
- Check your email for suspension notices
- Contact ElevenLabs support: support@elevenlabs.io
- Create new account if needed

### 4. **Trial Period Ended**
**Symptoms:** Worked during trial, stopped after expiry

**Solution:**
- Add payment method at: https://elevenlabs.io/app/billing
- Or create new trial account

### 5. **API Key Not Copied Correctly**
**Symptoms:** Random authentication errors

**Solution:**
- Copy key again (click the copy icon, don't type it)
- Make sure no extra spaces at start/end
- Verify key starts with `sk_` and is long (30+ chars)
- Use the "🔍 Test Key" button in the app

### 6. **Network/Firewall Issues**
**Symptoms:** Timeout or connection errors

**Solution:**
- Check your internet connection
- Disable VPN temporarily
- Check if firewall blocks ElevenLabs API
- Try from different network

---

## How to Get a Working API Key

### Option 1: Fix Current Account
1. Login to https://elevenlabs.io
2. Go to Settings → API Keys
3. Delete all existing keys
4. Create ONE new key
5. Copy it immediately
6. Test it using the test script:
   ```bash
   python test_elevenlabs.py
   ```

### Option 2: Create Fresh Account
1. Use a different email address
2. Sign up at https://elevenlabs.io
3. Verify your email
4. Go to Settings → API Keys
5. Create new key
6. Copy and test it

### Option 3: Use Paid Account
Free tier limitations:
- 10,000 characters/month
- May have rate limits
- Less reliable

Paid benefits:
- More characters
- Better rate limits
- Priority support
- More reliable

---

## Testing Your Key

### Method 1: In the App
1. Open Streamlit app
2. Go to Audio Generation mode
3. Paste your API key in sidebar
4. Click "🔍 Test Key" button
5. Should see "✅ API key valid!"

### Method 2: Test Script
```bash
python test_elevenlabs.py
```
Follow the prompts to validate and test generation.

### Method 3: Direct API Test
```python
from elevenlabs.client import ElevenLabs

api_key = "YOUR_KEY_HERE"
client = ElevenLabs(api_key=api_key)

# This should work if key is valid
voices = list(client.voices.get_all())
print(f"✅ Found {len(voices)} voices")
```

---

## Still Not Working?

### Check These:
- [ ] Using latest API key (not old/expired one)
- [ ] Account has available credits/quota
- [ ] Email is verified
- [ ] No spaces in pasted key
- [ ] Using correct key (not subscription ID or other value)
- [ ] Internet connection working
- [ ] Not blocked by firewall/VPN

### Get Help:
1. **ElevenLabs Support:** support@elevenlabs.io
2. **Check Status:** https://status.elevenlabs.io/
3. **Community:** https://discord.gg/elevenlabs

---

## Alternative: Use Different Service

If ElevenLabs continues to have issues, consider alternatives:
- Google Cloud Text-to-Speech
- Azure Cognitive Services
- Amazon Polly
- OpenAI TTS

(Would require code modifications)

---

## Important Notes

⚠️ **Never share your API key publicly**
- Treat it like a password
- Don't commit to GitHub
- Don't post in forums
- Regenerate if exposed

⚠️ **Free tier limitations**
- 10,000 chars/month = ~20-30 short generations
- Resets monthly
- Consider paid plan for heavy use

✅ **Best Practices**
- Generate new key for each project
- Delete unused keys
- Monitor your usage
- Keep key in environment variables (not in code)

---

## Success Checklist

After following this guide:
- [ ] Generated new API key
- [ ] Tested with test script - PASSED
- [ ] Tested in Streamlit app - PASSED
- [ ] Can generate audio successfully
- [ ] Saved working key securely

**If all checked:** You're good to go! 🎉
