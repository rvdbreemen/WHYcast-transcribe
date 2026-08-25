/*
  WHYcast web UI - the page's only script.

  It lives in a file rather than inline in base.html so the app can send a
  Content-Security-Policy with `script-src 'self'` and no 'unsafe-inline'.
  That header is what stops a hostile artifact from executing if it ever
  reaches the page's own DOM; an inline <script> would force us to weaken it
  to 'unsafe-inline', which weakens it for everything.
*/
(function () {
  'use strict';

  /* Timestamps arrive as epoch seconds (whycast.episodes.Artifact.mtime).
     Jinja has no date filter, so the server renders the raw number and this
     turns it into the viewer's local time. Without JavaScript the raw epoch
     stays visible - ugly, but honest and never wrong. */
  function renderTimes(root) {
    if (!root || !root.querySelectorAll) { return; }
    root.querySelectorAll('time[data-epoch]').forEach(function (el) {
      var seconds = Number(el.getAttribute('data-epoch'));
      if (!seconds) { return; }
      var when = new Date(seconds * 1000);
      el.textContent = when.toLocaleString();
      el.setAttribute('datetime', when.toISOString());
    });
  }

  /* Artifact tabs are plain links pointing at a sandboxed iframe (see
     episode.html). Marking the active one is presentation only - the
     navigation happens with or without this script. */
  function wireTabs() {
    var tabs = document.querySelectorAll('.tabs a.tab');
    if (!tabs.length) { return; }
    tabs.forEach(function (tab) {
      tab.addEventListener('click', function () {
        tabs.forEach(function (other) { other.removeAttribute('aria-current'); });
        tab.setAttribute('aria-current', 'true');
        var placeholder = document.getElementById('artifact-placeholder');
        if (placeholder) { placeholder.hidden = true; }
        var frame = document.getElementById('artifact-frame');
        if (frame) { frame.hidden = false; }
      });
    });
  }

  /* =====================================================================
     Jobs (ADR-008 phase 2, TASK-003)

     Everything below drives the queue dashboard and the job detail page.
     Three rules hold throughout and are not negotiable:

     * Job text - log lines, error messages, parameters - is written with
       textContent. Never innerHTML, never a template string assembled into
       markup. A pipeline error can contain any characters at all, and this is
       the layer where "no |safe in the templates" would otherwise be undone.
     * Class names built from server values go through oneOf(): a status or
       level is only ever one of the words we know.
     * The connection is never closed on a transport error for a job that is
       still running. EventSource reconnects by itself, resending
       Last-Event-ID, and webui.app resumes after that sequence number.
     ===================================================================== */

  var LOG_MAX_LINES = 2000;
  var LOG_LEVELS = ['debug', 'info', 'warning', 'error', 'critical'];
  var JOB_STATUSES = ['queued', 'running', 'succeeded', 'failed', 'cancelled'];

  /* A server value used as part of a class name must be one of ours. */
  function oneOf(value, allowed, fallback) {
    var text = String(value == null ? '' : value).toLowerCase();
    return allowed.indexOf(text) === -1 ? fallback : text;
  }

  function pad2(value) { return value < 10 ? '0' + value : String(value); }

  /* Mirrors the duration() macro in _macros.html. Two implementations of one
     format is a smell, but the alternative is a page that shows nothing until
     JavaScript runs. */
  function formatDuration(seconds) {
    var total = Math.floor(seconds);
    if (!isFinite(total) || total < 0) { return '-'; }
    if (total < 60) { return total + 's'; }
    if (total < 3600) { return Math.floor(total / 60) + 'm ' + pad2(total % 60) + 's'; }
    return Math.floor(total / 3600) + 'h ' + pad2(Math.floor((total % 3600) / 60)) + 'm';
  }

  /* "Running for 3m 12s" has to keep counting, or the number is a lie within
     a second of the page rendering. Only .live elements tick: a finished job's
     duration is a fact, not a clock. */
  function tickElapsed() {
    var now = Date.now() / 1000;
    document.querySelectorAll('.elapsed.live[data-since]').forEach(function (el) {
      var since = Number(el.getAttribute('data-since'));
      if (!since) { return; }
      el.textContent = formatDuration(now - since);
    });
  }

  /* ---- the live job log ------------------------------------------------ */

  function wireJobLog() {
    /* The same hooks the built-in fallback page in webui/app.py renders, so
       this works on both. Anything richer (progress bar, follow checkbox,
       outcome banner) is looked up separately and may legitimately be absent. */
    var box = document.querySelector('.joblog[data-job-id]');
    if (!box) { return; }
    var jobId = box.getAttribute('data-job-id');
    if (!jobId) { return; }
    var list = box.querySelector('[data-job-log]') || box;

    var nojs = document.getElementById('joblog-nojs');
    if (nojs) { nojs.hidden = true; }

    if (typeof window.EventSource === 'undefined') {
      note(list, 'This browser has no EventSource, so the log cannot be streamed.');
      return;
    }

    var url = box.getAttribute('data-events-url')
      || ('/api/jobs/' + encodeURIComponent(jobId) + '/events');
    /* A finished job's log cannot grow, so a dropped connection is the end of
       the story rather than something to wait out. Absent (the fallback page)
       means "assume it may still be running" - the safe direction. */
    var terminal = box.getAttribute('data-terminal') === 'true';

    var follow = document.getElementById('joblog-follow');
    var streamStatus = document.getElementById('joblog-status');
    /* By id for job_detail.html; the [data-job-progress] fallback is the
       attribute contract webui/app.py's built-in job page documents, so a
       progress bar rendered there is picked up too. Everything here is
       optional: only the container and the line list are required. */
    var bar = document.getElementById('job-progress')
      || document.querySelector('[data-job-progress]');
    var barLabel = document.getElementById('job-progress-label')
      || document.querySelector('[data-job-progress-label]');

    function following() { return !follow || follow.checked; }
    function atBottom() {
      return box.scrollHeight - box.scrollTop - box.clientHeight < 24;
    }
    function scrollDown() { if (following()) { box.scrollTop = box.scrollHeight; } }

    /* Scrolling up pauses the follow; scrolling back to the bottom resumes it.
       The checkbox stays the single source of truth so the state is visible
       and clickable, rather than a hidden mode. */
    box.addEventListener('scroll', function () {
      if (follow) { follow.checked = atBottom(); }
    });
    if (follow) {
      follow.addEventListener('change', function () { scrollDown(); });
    }

    function setStreamStatus(text) {
      if (streamStatus) { streamStatus.textContent = text; }
    }

    function setProgress(value) {
      if (typeof value !== 'number' || !isFinite(value)) { return; }
      var clamped = Math.max(0, Math.min(1, value));
      /* The event carries a fraction; the element decides its own scale. Read
         it from the element rather than assuming job_detail.html's max, or a
         <progress data-job-progress> rendered elsewhere with the default
         max="1" would sit pinned at 100% forever. */
      if (bar) { bar.value = clamped * (Number(bar.max) || 1); bar.hidden = false; }
      if (barLabel) {
        barLabel.textContent = Math.round(clamped * 100) + '%';
        barLabel.hidden = false;
      }
    }

    var seen = 0;
    function appendRecord(record) {
      if (!record || typeof record !== 'object') { return; }
      var li = document.createElement('li');
      li.className = 'logline lvl-' + oneOf(record.level, LOG_LEVELS, 'info');

      var when = document.createElement('span');
      when.className = 'logts';
      var ts = Number(record.ts);
      when.textContent = isFinite(ts) && ts > 0
        ? new Date(ts * 1000).toLocaleTimeString()
        : '';
      li.appendChild(when);

      if (record.step) {
        var step = document.createElement('span');
        step.className = 'logstep';
        step.textContent = String(record.step);
        li.appendChild(step);
      }

      var message = document.createElement('span');
      message.className = 'logmsg';
      /* textContent: this string came out of the pipeline. */
      message.textContent = record.message == null ? '' : String(record.message);
      li.appendChild(message);

      list.appendChild(li);
      seen += 1;
      while (list.children.length > LOG_MAX_LINES) {
        list.removeChild(list.firstChild);
      }
      setProgress(Number(record.progress));
      scrollDown();
    }

    var es = new EventSource(url);
    setStreamStatus('connecting...');

    es.onopen = function () {
      setStreamStatus(terminal ? 'replaying the log...' : 'connected - live');
    };

    /* The stream has no unnamed events: webui.app names every frame, so
       onmessage would never fire. */
    es.addEventListener('progress', function (event) {
      var record = parseJson(event.data);
      if (record) { appendRecord(record); }
    });

    es.addEventListener('status', function (event) {
      var job = parseJson(event.data);
      if (!job) { return; }
      applyJobStatus(job);
    });

    es.addEventListener('end', function (event) {
      var payload = parseJson(event.data) || {};
      es.close();
      applyJobStatus(payload);
      setStreamStatus(seen === 0
        ? 'stream closed - this job wrote no events'
        : 'stream closed - ' + seen + ' event' + (seen === 1 ? '' : 's'));
      note(list, 'End of log.');
      scrollDown();
    });

    /* Two different things arrive here. A frame the server named "error"
       carries data and means the server has given up; a transport failure has
       no data and is the browser telling us the connection dropped - which,
       for a running job, is exactly what EventSource retries by itself. */
    es.addEventListener('error', function (event) {
      if (event && typeof event.data === 'string' && event.data) {
        var payload = parseJson(event.data);
        var detail = (payload && payload.detail) ? String(payload.detail) : event.data;
        note(list, 'Stream error: ' + detail);
        setStreamStatus('stream error');
        es.close();
        return;
      }
      if (es.readyState === 2) {
        setStreamStatus('stream closed');
        return;
      }
      if (terminal) {
        es.close();
        setStreamStatus('disconnected; this job has finished, so nothing more is coming');
        return;
      }
      setStreamStatus('disconnected - reconnecting...');
    });
  }

  function parseJson(text) {
    if (typeof text !== 'string' || !text) { return null; }
    try { return JSON.parse(text); } catch (err) { return null; }
  }

  /* A grey italic line of our own, clearly not pipeline output. */
  function note(list, text) {
    if (!list) { return; }
    var li = document.createElement('li');
    li.className = 'logline meta';
    var span = document.createElement('span');
    span.className = 'logmsg';
    span.textContent = text;
    li.appendChild(span);
    list.appendChild(li);
  }

  /* Rewrites the status badge and the outcome banner from a streamed job row,
     so a job that finishes while you are watching it says so without a
     reload. */
  function applyJobStatus(job) {
    var status = job && job.status;
    if (!status) { return; }
    var known = oneOf(status, JOB_STATUSES, '');

    document.querySelectorAll('[data-job-status-badge]').forEach(function (el) {
      el.textContent = String(status);
      if (known && el.classList.contains('badge')) {
        el.className = 'badge s-' + known;
      }
    });

    var banner = document.querySelector('[data-job-outcome]');
    if (!banner) { return; }
    var tone = { succeeded: 'ok', failed: 'bad', cancelled: 'warn' }[known] || 'neutral';
    var text;
    if (known === 'succeeded') {
      text = 'Succeeded.';
    } else if (known === 'failed') {
      text = 'Failed';
      if (job.exit_code !== null && job.exit_code !== undefined) {
        text += ' (exit code ' + String(job.exit_code) + ')';
      }
      text += '.';
      if (job.error) { text += ' ' + String(job.error); }
    } else if (known === 'cancelled') {
      text = 'Cancelled.';
    } else if (known === 'running') {
      text = 'Running.';
    } else {
      text = 'Queued.';
    }
    banner.className = 'banner ' + tone;
    /* textContent, because job.error is pipeline text. */
    banner.textContent = text;

    /* A finished job is not still running for N seconds. */
    if (known !== 'running') {
      document.querySelectorAll('.elapsed.live').forEach(function (el) {
        el.classList.remove('live');
      });
    }
  }

  /* ---- enqueue and cancel ---------------------------------------------- */

  /*
    Both are plain <form method="post"> so they work with scripting off. Here
    they are intercepted, confirmed, and posted in the background, which is
    what turns "the browser navigates to a JSON document" into "a link to the
    job you just started".

    The confirmation text is rendered by the server (data-confirm) and names
    the job, the episode and whether it spends money. It is deliberately not
    assembled here: the template knows which episode you are looking at, and a
    warning built from the same catalogue the API validates against cannot
    drift from what will actually run.
  */
  function resultBox(form) {
    var id = form.getAttribute('data-result');
    return (id && document.getElementById(id)) || null;
  }

  function say(box, text, tone) {
    if (!box) { return; }
    box.className = 'job-result' + (tone ? ' ' + tone : '');
    box.textContent = text;
  }

  function describeError(res) {
    var detail = res && res.body ? res.body.detail : null;
    if (typeof detail === 'string' && detail) { return detail; }
    if (detail) { return JSON.stringify(detail); }
    return 'HTTP ' + (res ? res.status : '?');
  }

  function postForm(form) {
    return fetch(form.action, {
      method: 'POST',
      /* URLSearchParams posts application/x-www-form-urlencoded, which is what
         the same form would send without JavaScript - one body shape for both
         paths, and webui.app._job_request_payload() already reads it. */
      body: new URLSearchParams(new FormData(form)),
      headers: { 'Accept': 'application/json' },
      credentials: 'same-origin'
    }).then(function (response) {
      return response.text().then(function (text) {
        return {
          ok: response.ok,
          status: response.status,
          body: parseJson(text)
        };
      });
    });
  }

  function showJobLink(box, job) {
    if (!box) { return; }
    box.className = 'job-result';
    box.textContent = 'Queued ' + String(job.label || job.type || 'job') + ': ';
    var link = document.createElement('a');
    link.href = '/jobs/' + encodeURIComponent(String(job.id));
    link.textContent = String(job.id).slice(0, 8) + ' - open its live log';
    box.appendChild(link);
    if (job.cost) {
      box.appendChild(document.createTextNode(' '));
      var badge = document.createElement('span');
      badge.className = 'badge cost';
      badge.textContent = '$ paid API';
      box.appendChild(badge);
    }
  }

  function handleEnqueue(event, form) {
    event.preventDefault();
    var message = form.getAttribute('data-confirm');
    if (message && !window.confirm(message)) { return; }
    var box = resultBox(form);
    var button = form.querySelector('button');
    if (button) { button.disabled = true; }
    say(box, 'Queueing...', '');
    postForm(form).then(function (res) {
      if (!res.ok || !res.body || !res.body.id) {
        say(box, 'Could not queue this job: ' + describeError(res), 'bad');
        return;
      }
      showJobLink(box, res.body);
    }).catch(function (err) {
      say(box, 'Could not reach the server: ' + (err && err.message ? err.message : 'request failed'), 'bad');
    }).then(function () {
      if (button) { button.disabled = false; }
    });
  }

  function handleCancel(event, form) {
    event.preventDefault();
    var message = form.getAttribute('data-confirm');
    if (message && !window.confirm(message)) { return; }
    var button = form.querySelector('button');
    function restore() {
      if (button) { button.disabled = false; button.textContent = 'Cancel'; }
    }
    if (button) { button.disabled = true; button.textContent = 'cancelling...'; }
    postForm(form).then(function (res) {
      if (!res.ok) {
        restore();
        window.alert('Could not cancel this job: ' + describeError(res));
      }
      /* On success the button stays disabled and reads "cancelling...": a
         running job is only really cancelled once its process has stopped, and
         the dashboard's poll (or a reload) reports that when it happens. */
    }).catch(function (err) {
      restore();
      window.alert('Could not reach the server: ' + (err && err.message ? err.message : 'request failed'));
    });
  }

  /* Delegated from the document so htmx swaps (the dashboard replaces its own
     live region every few seconds) need no re-wiring. */
  function wireJobForms() {
    document.addEventListener('submit', function (event) {
      var form = event.target;
      if (!form || typeof form.matches !== 'function') { return; }
      if (form.matches('form[data-enqueue]')) {
        handleEnqueue(event, form);
      } else if (form.matches('form[data-cancel]')) {
        handleCancel(event, form);
      }
    });
  }

  function start() {
    renderTimes(document);
    wireTabs();
    wireJobLog();
    wireJobForms();
    tickElapsed();
    window.setInterval(tickElapsed, 1000);
    /* Re-render over the whole document, not event.target: an outerHTML swap
       detaches its target, so a handler bound to that node would miss the new
       rows. htmx:load is the documented new-content hook; afterSwap is kept as
       a belt-and-braces second trigger. 62 rows cost nothing. */
    function refresh() { renderTimes(document); tickElapsed(); }
    document.body.addEventListener('htmx:load', refresh);
    document.body.addEventListener('htmx:afterSwap', refresh);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', start);
  } else {
    start();
  }
})();
