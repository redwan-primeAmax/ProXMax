"""এক cell-এর দৃশ্যমান form; credential কোনো Colab @param/widget state নয়।"""
import json

# A native password input, not a @param (which writes into notebook source) or
# a synced ipywidget (whose value can be included in saved widget state).
_FORM_JS = r"""
new Promise((resolve) => {
  const needsKey = __PROXMAX_NEEDS_KEY__;
  const model = __PROXMAX_MODEL__;
  const old = document.getElementById('proxmax-controls');
  if (old) old.remove();
  const root = document.createElement('section');
  root.id = 'proxmax-controls';
  root.innerHTML = `
    <style>
      #proxmax-controls { font: 14px/1.6 system-ui,sans-serif; max-width: 640px;
        padding: 24px; border: 1px solid #d4dce6; border-radius: 14px;
        background: #f8fafc; color: #172b4d; box-sizing: border-box; }
      #proxmax-controls h3 { margin: 0 0 4px; font-size: 21px; }
      #proxmax-controls p { margin: 4px 0 18px; color: #475569; }
      #proxmax-controls .field { display: block; margin: 16px 0; font-weight: 600; }
      #proxmax-controls select, #proxmax-controls input[type=password] {
        display: block; width: 100%; box-sizing: border-box; margin-top: 6px;
        padding: 11px 12px; border: 1px solid #aebdce; border-radius: 8px;
        background: white; color: #172b4d; font: inherit; }
      #proxmax-controls button { padding: 11px 16px; margin: 16px 8px 4px 0;
        border: 0; border-radius: 8px; background: #1d4ed8; color: white;
        font: inherit; font-weight: 600; cursor: pointer; }
      #proxmax-controls button[type=button] { background: #e2e8f0; color: #172b4d; }
      #proxmax-controls button:disabled { opacity: .55; cursor: default; }
      #proxmax-controls summary { cursor: pointer; }
      #proxmax-controls small { display: block; color: #475569; }
      #proxmax-controls [role=status] { margin-top: 12px; font-weight: 600; }
      #proxmax-controls :focus-visible { outline: 3px solid #60a5fa; outline-offset: 2px; }
    </style>
    <h3>ProXMax · ইংরেজি → বাংলা</h3>
    <p>Model: <strong id="proxmax-model"></strong></p>
    <form autocomplete="off">
      <label class="field" for="proxmax-domain">ভিডিওর ধরন (Domain)
        <select id="proxmax-domain" name="domain">
          <option value="general">সাধারণ — general</option>
          <option value="football">ফুটবল / FC Mobile — football</option>
        </select>
      </label>
      <label class="field" for="proxmax-api-key">Gemini API key
        <input id="proxmax-api-key" type="password" autocomplete="off"
          spellcheck="false" autocapitalize="none" required maxlength="256"
          placeholder="AI Studio থেকে পাওয়া key এখানে paste করুন">
      </label>
      <small>Key masked থাকবে; source বা output-এ লেখা হবে না। Start/বাতিল করলে ঘরটি খালি হবে।</small>
      <p><small id="proxmax-privacy">ইংরেজি transcript Gemini-তে যাবে; quota/চার্জ প্রযোজ্য হতে পারে।</small></p>
      <details><summary>Drive-এর অতিরিক্ত সেটিংস</summary>
        <label><input id="proxmax-remount" type="checkbox"> নষ্ট Drive সংযোগ নতুন করে mount করুন</label>
      </details>
      <button type="submit">সেটআপ ও অনুবাদ চালু করুন</button>
      <button type="button" id="proxmax-cancel">বাতিল</button>
      <div role="status" aria-live="polite" id="proxmax-status">Domain বেছে key paste করুন।</div>
    </form>`;
  document.body.appendChild(root);
  const form = root.querySelector('form');
  const key = root.querySelector('#proxmax-api-key');
  const status = root.querySelector('#proxmax-status');
  root.querySelector('#proxmax-model').textContent = model;
  if (!needsKey) {
    key.required = false;
    key.disabled = true;
    key.placeholder = 'Local অনুবাদে API key লাগবে না';
    root.querySelector('#proxmax-privacy').textContent = 'Local অনুবাদ: transcript Gemini-তে পাঠানো হবে না।';
    status.textContent = 'Domain বেছে চালু করুন।';
  }
  let submitted = false;
  const resize = () => {
    if (typeof google !== 'undefined' && google.colab && google.colab.output) {
      google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);
    }
  };
  const disable = () => {
    key.value = '';
    form.querySelectorAll('input, select, button').forEach(el => { el.disabled = true; });
  };
  form.addEventListener('submit', event => {
    event.preventDefault();
    if (submitted) return;
    if (needsKey && !/^[A-Za-z0-9_-]{16,256}$/.test(key.value.trim())) {
      status.textContent = 'সঠিক API key paste করুন; key চ্যাটে দেবেন না।';
      key.focus();
      resize();
      return;
    }
    submitted = true;
    const result = {
      domain: root.querySelector('#proxmax-domain').value,
      api_key: needsKey ? key.value.trim() : '',
      remount: root.querySelector('#proxmax-remount').checked
    };
    disable();
    status.textContent = 'সেটআপ ও অনুবাদ চলছে… অগ্রগতি নিচে দেখুন।';
    resize();
    resolve(result);
  });
  root.querySelector('#proxmax-cancel').addEventListener('click', () => {
    if (submitted) return;
    submitted = true;
    disable();
    status.textContent = 'বাতিল হয়েছে। আবার শুরু করতে cell-এর ▶ চাপুন।';
    resize();
    resolve(null);
  });
  root.querySelector('details').addEventListener('toggle', resize);
  resize();
})
"""


def read_form(require_api_key=True, model="gemini-3.5-flash-lite"):
    """Wait inside this cell; do not display/print the returned credential object."""
    from google.colab import output
    script = _FORM_JS.replace("__PROXMAX_NEEDS_KEY__", json.dumps(require_api_key))
    script = script.replace("__PROXMAX_MODEL__", json.dumps(model))
    value = output.eval_js(script)
    if value is None:
        raise KeyboardInterrupt
    if (not isinstance(value, dict) or value.get("domain") not in ("general", "football")
            or not isinstance(value.get("api_key"), str)
            or (require_api_key and not value["api_key"].strip())
            or type(value.get("remount")) is not bool):
        if isinstance(value, dict):
            value.clear()
        raise ValueError("Form-এর তথ্য সঠিক নয়; cell আবার চালান।")
    return value


def finish_form(status):
    from google.colab import output
    message = {"complete": "সম্পন্ন। ফলাফল ও review নির্দেশনা নিচে দেখুন।",
               "failed": "কাজ সম্পন্ন হয়নি। নিচের error দেখুন; আবার cell চালাতে পারেন।",
               "cancelled": "কাজ বন্ধ হয়েছে। আবার শুরু করতে cell-এর ▶ চাপুন।"}.get(status, "কাজ শেষ।")
    try:
        output.eval_js("(() => { const el = document.getElementById('proxmax-status'); "
                       "if (el) el.textContent = " + json.dumps(message) + "; })()")
    except Exception:
        # Cosmetic status must not hide a real worker result/error; terminal is authoritative.
        pass
