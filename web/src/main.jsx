import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  BarChart3,
  BrainCircuit,
  CheckCircle2,
  CircleDot,
  Cpu,
  FlaskConical,
  Gauge,
  Layers3,
  ListChecks,
  Pause,
  Play,
  RefreshCw,
  Route,
  Server,
  Sparkles,
  TrendingUp,
} from "lucide-react";
import "./styles.css";

const formatter = new Intl.NumberFormat("en-US", {
  maximumFractionDigits: 4,
  minimumFractionDigits: 3,
});

function App() {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  const [paused, setPaused] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState(null);
  const [lastRefresh, setLastRefresh] = useState(null);

  const load = async () => {
    try {
      const response = await fetch("/api/runs", { cache: "no-store" });
      if (!response.ok) throw new Error(`API returned ${response.status}`);
      const payload = await response.json();
      setData(payload);
      setError("");
      setLastRefresh(new Date());
      setSelectedIndex((current) => {
        if (current !== null) return current;
        return payload.events.at(-1)?.index ?? null;
      });
    } catch (err) {
      setError(err.message);
    }
  };

  useEffect(() => {
    load();
  }, []);

  useEffect(() => {
    if (paused) return undefined;
    const timer = window.setInterval(load, 2500);
    return () => window.clearInterval(timer);
  }, [paused]);

  const events = data?.events ?? [];
  const claudeRuns = data?.claudeRuns ?? [];
  const selected = useMemo(
    () => events.find((event) => event.index === selectedIndex) ?? events.at(-1),
    [events, selectedIndex],
  );
  const goal = useMemo(() => extractGoal(data?.program ?? ""), [data]);
  const lossStats = useMemo(() => summarizeLoss(events), [events]);
  const changeInsights = useMemo(() => summarizeChanges(events), [events]);
  const selectExperiment = (index) => {
    setSelectedIndex(index);
    window.requestAnimationFrame(() => {
      document.getElementById("selected-candidate")?.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  };

  return (
    <main className="appShell">
      <aside className="sidebar">
        <div className="brandMark">
          <BrainCircuit size={25} />
          <div>
            <strong>Autoresearch BCI</strong>
            <span>EEG control research</span>
          </div>
        </div>

        <nav className="sideNav" aria-label="dashboard sections">
          <a className="active" href="#progress"><BarChart3 size={17} /> Progress</a>
          <a href="#best"><Activity size={17} /> Current best</a>
          <a href="#runs"><Server size={17} /> Runs</a>
        </nav>

        <div className="sideBlock">
          <span className="sideLabel">Active branch</span>
          <strong>#{data?.stats.bestIndex ?? 1} is the parent</strong>
          <p>Loss is normalized from reward so the best run is {formatLoss(lossStats.bestLoss)} and lower is better.</p>
        </div>

        <div className="sideBlock">
          <span className="sideLabel">Compute</span>
          <strong>{data?.stats.modalCount ?? 0} Modal CPU rows</strong>
          <p>{data?.stats.localCount ?? 0} local candidates, {data?.stats.claudeCount ?? 0} Claude Code agent rows.</p>
        </div>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div>
            <div className="eyebrow">
              <CircleDot size={14} />
              {paused ? "Polling paused" : "Live autoresearch"}
            </div>
            <h1>Autoresearch BCI</h1>
            <p>{goal}</p>
          </div>
          <div className="toolbar" aria-label="dashboard controls">
            <button type="button" className="iconButton" onClick={() => setPaused((value) => !value)} title={paused ? "Resume live polling" : "Pause live polling"}>
              {paused ? <Play size={18} /> : <Pause size={18} />}
            </button>
            <button type="button" className="iconButton" onClick={load} title="Refresh now">
              <RefreshCw size={18} />
            </button>
            <span className={paused ? "status paused" : "status live"}>
              <CircleDot size={14} />
              {paused ? "Paused" : "Live"}
            </span>
          </div>
        </header>

        {error ? <div className="notice">{error}</div> : null}

        <section className="liveTicker" aria-label="live run status">
          <strong>LIVE</strong>
          <span>LOSS BEST: {formatLoss(lossStats.bestLoss)}</span>
          <span>LOSS WORST: {formatLoss(lossStats.worstLoss)}</span>
          <span>CANDIDATES: {data?.stats.totalCandidates ?? 0}</span>
          <span>MODAL CPU: {data?.stats.modalCount ?? 0}</span>
          <span>CLAUDE: {data?.stats.claudeCount ?? 0}</span>
          <span>UPDATED: {lastRefresh ? lastRefresh.toLocaleTimeString() : "waiting"}</span>
        </section>

        <section className="metrics" aria-label="run summary">
          <Metric icon={<FlaskConical />} label="Candidates" value={data?.stats.totalCandidates ?? 0} />
          <Metric icon={<CheckCircle2 />} label="Kept" value={data?.stats.acceptedCount ?? 0} />
          <Metric icon={<Cpu />} label="Claude Agent" value={data?.stats.claudeCount ?? 0} />
          <Metric icon={<Gauge />} label="Best Loss" value={formatLoss(lossStats.bestLoss)} />
          <Metric icon={<TrendingUp />} label="Worst Loss" value={formatLoss(lossStats.worstLoss)} />
        </section>

        <section className="dashboardGrid" id="progress">
          <div className="panel chartPanel">
            <PanelTitle icon={<BarChart3 />} title="Ratchet Step Graph" meta={`${events.length} candidates`} />
            <ChartNarrative events={events} />
            <RewardChart events={events} onSelect={selectExperiment} selectedIndex={selected?.index} />
          </div>

          <div className="panel" id="best">
            <PanelTitle icon={<Activity />} title="Current Best" meta={data?.best?.subject ? `${data.best.subject} stage ${data.best.stage}` : "waiting"} />
            <BestSummary best={data?.best} />
          </div>

          <div className="panel">
            <PanelTitle icon={<Route />} title="Improvement Chain" meta={`${data?.lineage?.length ?? 0} kept`} />
            <RatchetChain lineage={data?.lineage ?? []} best={data?.best} />
          </div>

          <div className="panel">
            <PanelTitle icon={<Sparkles />} title="What Helped" meta="loss drops" />
            <InsightList insights={changeInsights} onSelect={selectExperiment} />
          </div>

          <div className="panel">
            <PanelTitle icon={<ListChecks />} title="Best Candidates" meta="lowest loss" />
            <BestCandidateList events={events} onSelect={selectExperiment} />
          </div>

          <div className="panel">
            <PanelTitle icon={<Cpu />} title="Agent Updates" meta="latest" />
            <AgentUpdates events={events} onSelect={selectExperiment} />
          </div>

          <div className="panel">
            <PanelTitle icon={<BrainCircuit />} title="Claude Code Research" meta={`${claudeRuns.length} rows`} />
            <ClaudeRuns runs={claudeRuns} />
          </div>
        </section>

        <section className="dashboardGrid lower" id="runs">
          <div className="panel">
            <PanelTitle icon={<Server />} title="Candidate Runs" meta={lastRefresh ? `refreshed ${lastRefresh.toLocaleTimeString()}` : "loading"} />
            <RunTable events={events} selectedIndex={selected?.index} onSelect={selectExperiment} />
          </div>

          <div className="panel selectedPanel" id="selected-candidate">
            <PanelTitle icon={<FlaskConical />} title="Selected Candidate" meta={selected ? `#${selected.index}` : "none"} />
            <CandidateDetails event={selected} />
          </div>
        </section>
      </section>
    </main>
  );
}

function Metric({ icon, label, value }) {
  return (
    <div className="metric">
      <div className="metricIcon">{icon}</div>
      <div>
        <span>{label}</span>
        <strong>{value}</strong>
      </div>
    </div>
  );
}

function PanelTitle({ icon, title, meta }) {
  return (
    <div className="panelTitle">
      <div>
        {icon}
        <h2>{title}</h2>
      </div>
      <span>{meta}</span>
    </div>
  );
}

function RewardChart({ events, onSelect, selectedIndex }) {
  if (!events.length) {
    return <div className="empty">Run `python3 loop.py` or `modal run modal_app.py` to populate the graph.</div>;
  }

  const width = 1480;
  const height = 660;
  const pad = { top: 82, right: 70, bottom: 88, left: 104 };
  const rankedBase = [...events].sort((a, b) => (b.validationLoss ?? 0) - (a.validationLoss ?? 0) || a.index - b.index);
  const rankedEvents = rankedBase.map((event, rank) => ({
    ...event,
    rank,
    rankLossDrop: rank === 0 ? null : (rankedBase[rank - 1]?.validationLoss ?? 0) - (event.validationLoss ?? 0),
  }));
  const losses = rankedEvents.map((event) => event.validationLoss ?? 0);
  const keptCount = events.filter((event) => event.acceptedByLog).length;
  const maxLoss = Math.max(...losses);
  const minLoss = Math.min(...losses);
  const yPadding = Math.max(0.015, (maxLoss - minLoss) * 0.08);
  const yMin = minLoss - yPadding;
  const yMax = maxLoss + yPadding;
  const x = (idx) => pad.left + (idx / Math.max(1, rankedEvents.length - 1)) * (width - pad.left - pad.right);
  const y = (value) => {
    const clamped = Math.max(yMin, Math.min(yMax, value));
    return height - pad.bottom - ((clamped - yMin) / Math.max(0.0001, yMax - yMin)) * (height - pad.top - pad.bottom);
  };
  const xTicks = makeXTicks(rankedEvents.length);
  const yTicks = makeYTicks(yMin, yMax, 6);
  const bestPath = makeStepPath(rankedEvents.map((event) => event.validationLoss ?? 0), x, y);
  const labels = makeAnnotations(rankedEvents);
  const bestRanked = rankedEvents.at(-1);

  return (
    <svg className="chart progressChart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="autoresearch improvement ladder ranked worst to best">
      <rect className="plotBackground" x="0" y="0" width={width} height={height} />
      <text className="plotTitle" x={width / 2} y="34" textAnchor="middle">
        {`Validation Loss Ladder: ${events.length} Experiments Ranked Worst → Best, ${keptCount} Ratchet Keeps`}
      </text>
      <text className="plotSubtitle" x={width / 2} y="58" textAnchor="middle">
        Lower is better. Each step shows the next observed loss improvement; labels are clickable.
      </text>
      {yTicks.map((tick) => (
        <g key={`y-${tick}`}>
          <line className="plotGrid" x1={pad.left} x2={width - pad.right} y1={y(tick)} y2={y(tick)} />
          <text className="axisTick" x={pad.left - 16} y={y(tick) + 5} textAnchor="end">
            {formatLoss(tick)}
          </text>
        </g>
      ))}
      {xTicks.map((tick) => (
        <g key={`x-${tick}`}>
          <line className="plotGrid" x1={x(tick)} x2={x(tick)} y1={pad.top} y2={height - pad.bottom} />
          <text className="axisTick" x={x(tick)} y={height - pad.bottom + 34} textAnchor="middle">
            {tick === 0 ? "worst" : tick === rankedEvents.length - 1 ? "best" : tick + 1}
          </text>
        </g>
      ))}
      <line className="axisLine" x1={pad.left} y1={height - pad.bottom} x2={width - pad.right} y2={height - pad.bottom} />
      <line className="axisLine" x1={pad.left} y1={pad.top} x2={pad.left} y2={height - pad.bottom} />
      <text className="axisLabel" x={width / 2} y={height - 22} textAnchor="middle">Experiments ranked by validation loss (worst → best)</text>
      <text className="axisLabel" x={-height / 2} y="32" textAnchor="middle" transform="rotate(-90)">Validation loss (lower is better)</text>

      {rankedEvents.map((event, idx) => {
        if (idx === 0) return null;
        const previous = rankedEvents[idx - 1]?.validationLoss ?? event.validationLoss ?? 0;
        return (
          <line
            className="candidateStem improvementStem"
            key={`stem-${event.index}-${idx}`}
            x1={x(idx)}
            x2={x(idx)}
            y1={y(previous)}
            y2={y(event.validationLoss ?? previous)}
          />
        );
      })}
      <path className="runningBestPath" d={bestPath} />
      {rankedEvents.map((event, idx) => (
        <g key={`${event.signature}-${event.index}`} className="pointGroup" onClick={() => onSelect(event.index)} tabIndex="0" role="button">
          <circle
            className={`${event.index === bestRanked?.index ? "plotPoint keptPoint" : "plotPoint discardedPoint"} ${event.index === selectedIndex ? "selectedPoint" : ""}`}
            cx={x(idx)}
            cy={y(event.validationLoss ?? 0)}
            r={event.index === bestRanked?.index ? 7 : 3.8}
          />
          <title>{`rank ${idx + 1}: experiment #${event.index} loss ${formatLoss(event.validationLoss)} ${shortHypothesis(event)}`}</title>
        </g>
      ))}
      {labels.map(({ event, text }, labelIdx) => {
        const idx = event.rank;
        const pointX = x(idx);
        const pointY = y(event.validationLoss ?? 0);
        const textX = Math.max(pad.left + 12, Math.min(width - pad.right - 260, pointX + (idx > rankedEvents.length * 0.68 ? -260 : 20)));
        const textY = Math.max(pad.top + 18, Math.min(height - pad.bottom - 38, pointY + (labelIdx % 2 === 0 ? -56 : 44)));
        const labelWidth = Math.min(250, Math.max(108, text.length * 7.2 + 18));
        return (
          <g
            className={`annotationGroup ${event.index === selectedIndex ? "selectedAnnotation" : ""}`}
            key={`label-${event.index}`}
            onClick={() => onSelect(event.index)}
            onKeyDown={(keyboardEvent) => {
              if (keyboardEvent.key === "Enter" || keyboardEvent.key === " ") {
                keyboardEvent.preventDefault();
                onSelect(event.index);
              }
            }}
            tabIndex="0"
            role="button"
            aria-label={`Select experiment ${event.index}: ${text}`}
          >
            <line className="annotationLeader" x1={pointX} y1={pointY} x2={textX + 8} y2={textY + 8} />
            <rect x={textX} y={textY - 15} width={labelWidth} height="25" rx="5" />
            <text className="annotationLabel" x={textX + 9} y={textY + 2}>{text}</text>
          </g>
        );
      })}
      <g className="plotLegend" transform={`translate(${width - 188} 60)`}>
        <rect width="158" height="88" rx="4" />
        <circle className="legendDiscarded" cx="18" cy="22" r="4" />
        <text x="42" y="27">Candidate</text>
        <circle className="legendKept" cx="18" cy="48" r="7" />
        <text x="42" y="53">Best</text>
        <line className="legendBest" x1="10" x2="30" y1="72" y2="72" />
        <text x="42" y="77">Loss ladder</text>
      </g>
    </svg>
  );
}

function ChartNarrative({ events }) {
  if (!events.length) return null;
  const ranked = [...events].sort((a, b) => (b.validationLoss ?? 0) - (a.validationLoss ?? 0) || a.index - b.index);
  const worst = ranked[0];
  const best = ranked.at(-1);
  const jumps = ranked.slice(1).map((event, idx) => ({
    event,
    from: ranked[idx],
    jump: (ranked[idx]?.validationLoss ?? 0) - (event.validationLoss ?? 0),
  }));
  const biggestJump = jumps.sort((a, b) => b.jump - a.jump)[0];
  return (
    <div className="chartNarrative">
      <div>
        <span>Worst observed</span>
        <strong>#{worst.index} · {shortHypothesis(worst)}</strong>
      </div>
      <div>
        <span>Biggest loss drop</span>
        <strong>{biggestJump ? `#${biggestJump.from.index} → #${biggestJump.event.index} · -${formatLoss(biggestJump.jump)}` : "waiting"}</strong>
      </div>
      <div>
        <span>Best pipeline</span>
        <strong>#{best.index} · loss {formatLoss(best.validationLoss)}</strong>
      </div>
    </div>
  );
}

function makeStepPath(values, x, y) {
  if (!values.length) return "";
  let d = `M ${x(0).toFixed(2)} ${y(values[0]).toFixed(2)}`;
  for (let idx = 1; idx < values.length; idx += 1) {
    d += ` H ${x(idx).toFixed(2)} V ${y(values[idx]).toFixed(2)}`;
  }
  return d;
}

function makeXTicks(count) {
  if (count <= 1) return [0];
  const max = count - 1;
  const step = count > 60 ? 20 : count > 30 ? 10 : 5;
  const ticks = [];
  for (let tick = 0; tick <= max; tick += step) ticks.push(tick);
  if (ticks.at(-1) !== max) ticks.push(max);
  return ticks;
}

function makeYTicks(min, max, count) {
  const span = Math.max(0.001, max - min);
  const rawStep = span / Math.max(1, count - 1);
  const magnitude = 10 ** Math.floor(Math.log10(rawStep));
  const normalized = rawStep / magnitude;
  const nice = normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 5 ? 5 : 10;
  const step = nice * magnitude;
  const start = Math.floor(min / step) * step;
  const ticks = [];
  for (let value = start; value <= max + step * 0.5; value += step) {
    if (value >= min - step * 0.5) ticks.push(value);
  }
  return ticks.slice(-8);
}

function makeAnnotations(events) {
  const jumps = events.slice(1).map((event, idx) => ({
    event,
    jump: (events[idx]?.validationLoss ?? 0) - (event.validationLoss ?? 0),
  }));
  const biggest = jumps.sort((a, b) => b.jump - a.jump).slice(0, 5).map(({ event, jump }) => ({
    event,
    text: `-${formatLoss(jump)} ${shortHypothesis(event)}`,
  }));
  const endpoints = [
    { event: events[0], text: `worst: ${shortHypothesis(events[0])}` },
    { event: events.at(-1), text: `best: ${shortHypothesis(events.at(-1))}` },
  ];
  return [...endpoints, ...biggest].filter(({ event }) => event);
}

function shortHypothesis(event) {
  const config = event.config ?? {};
  if ((event.hypothesis ?? "").includes("channel")) return `channels ${config.channels?.join("") ?? ""}`;
  if ((event.hypothesis ?? "").includes("window")) return `window ${config.window_start ?? 0}-${(config.window_start ?? 0) + (config.window_size ?? 0)}`;
  if ((event.hypothesis ?? "").includes("classifier") || (event.hypothesis ?? "").includes("optimizer")) return `${config.model ?? "model"} lr ${config.learning_rate ?? ""}`;
  if ((event.hypothesis ?? "").includes("feature")) return `${(config.features ?? []).slice(0, 3).join("+")}`;
  if ((event.hypothesis ?? "").includes("clipping")) return `clip ${config.clip}`;
  if ((event.hypothesis ?? "").includes("smoothing")) return `smooth ${config.smooth}`;
  return (event.hypothesis ?? "candidate").slice(0, 42);
}

function BestSummary({ best }) {
  if (!best) return <div className="empty">No best config found yet.</div>;
  const config = best.config ?? {};
  return (
    <div className="bestSummary">
      <div className="bigReward">0.0000</div>
      <div className="kvGrid">
        <KV label="Validation Loss" value="best observed" />
        <KV label="Reward" value={format(best.reward)} />
        <KV label="Model" value={config.model} />
        <KV label="Features" value={(config.features ?? []).join(", ")} />
        <KV label="Channels" value={(config.channels ?? []).join(" ")} />
        <KV label="Window" value={`${config.window_start ?? 0}-${(config.window_start ?? 0) + (config.window_size ?? 0)} samples`} />
        <KV label="Clip" value={config.clip} />
        <KV label="Smooth" value={config.smooth} />
      </div>
    </div>
  );
}

function InsightList({ insights, onSelect }) {
  if (!insights.length) return <div className="empty">No candidate changes yet.</div>;
  return (
    <div className="insightList">
      {insights.slice(0, 6).map((item) => (
        <button type="button" key={item.key} onClick={() => onSelect(item.best.index)}>
          <span>{item.label}</span>
          <strong>{formatLoss(item.best.validationLoss)} loss</strong>
          <small>best #{item.best.index} · {item.count} tries · avg {formatLoss(item.averageLoss)}</small>
        </button>
      ))}
    </div>
  );
}

function BestCandidateList({ events, onSelect }) {
  const best = [...events].sort((a, b) => (a.validationLoss ?? 0) - (b.validationLoss ?? 0) || a.index - b.index).slice(0, 6);
  if (!best.length) return <div className="empty">No candidates yet.</div>;
  return (
    <div className="bestList">
      {best.map((event, idx) => (
        <button type="button" key={`${event.signature}-${event.index}`} onClick={() => onSelect(event.index)}>
          <span>#{idx + 1}</span>
          <div>
            <strong>Experiment #{event.index}</strong>
            <small>{shortHypothesis(event)}</small>
          </div>
          <b>{formatLoss(event.validationLoss)}</b>
        </button>
      ))}
    </div>
  );
}

function AgentUpdates({ events, onSelect }) {
  const recent = [...events].slice(-5).reverse();
  if (!recent.length) return <div className="empty">Waiting for candidates.</div>;
  return (
    <div className="updatesList">
      {recent.map((event) => (
        <button type="button" key={`${event.signature}-${event.index}`} onClick={() => onSelect(event.index)}>
          <span className={lossDeltaClass(event.deltaLossFromParent)}>{event.deltaLossFromParent === null ? "base" : signedLossDelta(event.deltaLossFromParent)}</span>
          <div>
            <strong>Experiment #{event.index}</strong>
            <small>{shortHypothesis(event)} · loss {formatLoss(event.validationLoss)}</small>
          </div>
        </button>
      ))}
    </div>
  );
}

function ClaudeRuns({ runs }) {
  const recent = [...runs].slice(-6).reverse();
  if (!recent.length) return <div className="empty">No Claude Code agent rows yet.</div>;
  return (
    <div className="updatesList">
      {recent.map((run) => (
        <div key={`${run.tag}-${run.iteration}`} className="updateCard">
          <span className={run.status === "keep" ? "positive" : run.status === "crash" ? "negative" : "neutral"}>
            {run.status}
          </span>
          <div>
            <strong>{run.tag} · iteration {run.iteration}</strong>
            <small>reward {format(run.reward)} · {run.seconds?.toFixed?.(1) ?? run.seconds}s · {run.description}</small>
          </div>
        </div>
      ))}
    </div>
  );
}

function RatchetChain({ lineage, best }) {
  if (!lineage.length) {
    return <div className="empty">No kept improvements logged yet.</div>;
  }
  const latest = lineage.at(-1);
  return (
    <div className="chain">
      <div className="chainIntro">
        <strong>Active search branch</strong>
        <span>Every new candidate is proposed from the latest kept pipeline until one beats {format(best?.reward)}.</span>
      </div>
      <ol>
        {lineage.map((event) => (
          <li key={`${event.signature}-${event.index}`}>
            <div className="chainDot" />
            <div>
              <strong>#{event.index} kept at {format(event.reward)}</strong>
              <span>
                {event.parentIndex ? `Built from #${event.parentIndex}` : "Bootstrap best from the first logged sweep"}
                {event.deltaFromParent !== null ? `, ${signed(event.deltaFromParent)} reward` : ""}
              </span>
              <p>{shortHypothesis(event)}</p>
            </div>
          </li>
        ))}
      </ol>
      <div className="nextStep">
        <span>Next step</span>
        <strong>Beat #{latest.index} above {format(latest.reward)}</strong>
      </div>
    </div>
  );
}

function RunTable({ events, selectedIndex, onSelect }) {
  if (!events.length) return <div className="empty">No run rows yet.</div>;
  return (
    <div className="tableWrap">
      <table>
        <thead>
          <tr>
            <th>Step</th>
            <th>Loss</th>
            <th>Δ Loss</th>
            <th>Heldout</th>
            <th>Gap</th>
            <th>Backend</th>
            <th>What changed</th>
          </tr>
        </thead>
        <tbody>
          {[...events].reverse().map((event) => (
            <tr
              key={`${event.signature}-${event.index}`}
              className={event.index === selectedIndex ? "selectedRow" : ""}
              onClick={() => onSelect(event.index)}
            >
              <td>{event.acceptedByLog ? <CheckCircle2 size={15} /> : null} #{event.index}</td>
              <td>{formatLoss(event.validationLoss)}</td>
              <td className={lossDeltaClass(event.deltaLossFromParent)}>{event.deltaLossFromParent === null ? "base" : signedLossDelta(event.deltaLossFromParent)}</td>
              <td>{format(event.heldout?.closed_loop_score)}</td>
              <td>{format(event.generalization_gap)}</td>
              <td><span className={event.backend === "modal-cpu" ? "pill modal" : "pill"}>{event.backend ?? "local"}</span></td>
              <td>
                <strong className="changeLabel">{shortHypothesis(event)}</strong>
                <span className="changeSub">{event.hypothesis}</span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function CandidateDetails({ event }) {
  if (!event) return <div className="empty">Select a row or graph point.</div>;
  return (
    <div className="details">
      <p className="hypothesis">{event.hypothesis}</p>
      <div className="kvGrid compact">
        <KV label="Validation Loss" value={formatLoss(event.validationLoss)} />
        <KV label="Δ Loss" value={event.deltaLossFromParent === null ? "bootstrap" : signedLossDelta(event.deltaLossFromParent)} />
        <KV label="Reward" value={format(event.reward)} />
        <KV label="Running Best" value={format(event.runningBest)} />
        <KV label="Parent Best" value={event.parentBest === null ? "bootstrap" : `#${event.parentIndex} at ${format(event.parentBest)}`} />
        <KV label="Delta" value={event.deltaFromParent === null ? "n/a" : signed(event.deltaFromParent)} />
        <KV label="Closed Loop" value={format(event.heldout?.closed_loop_score)} />
        <KV label="Balanced Acc" value={format(event.heldout?.balanced_accuracy)} />
        <KV label="Macro F1" value={format(event.heldout?.macro_f1)} />
        <KV label="CV Score" value={format(event.cv?.closed_loop_score)} />
        <KV label="Gap" value={format(event.generalization_gap)} />
        <KV label="Seconds" value={format(event.seconds)} />
        <KV label="Backend" value={event.backend ?? "local"} />
        <KV label="Signature" value={event.signature} />
      </div>
      <pre>{JSON.stringify(event.config, null, 2)}</pre>
    </div>
  );
}

function KV({ label, value }) {
  return (
    <div className="kv">
      <span>{label}</span>
      <strong>{value ?? "n/a"}</strong>
    </div>
  );
}

function extractGoal(program) {
  const line = program.split(/\r?\n/).find((entry) => entry.startsWith("GOAL:"));
  return line ? line.replace("GOAL:", "").trim() : "Live view of autoresearch candidate evaluation and ratcheting.";
}

function format(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
  return formatter.format(Number(value));
}

function formatLoss(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
  return formatter.format(Math.max(0, Number(value)));
}

function signed(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
  if (Math.abs(Number(value)) < 1e-9) return "0.000";
  const sign = Number(value) >= 0 ? "+" : "";
  return `${sign}${format(value)}`;
}

function signedLossDelta(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
  if (Math.abs(Number(value)) < 1e-9) return "0.000";
  const sign = Number(value) > 0 ? "+" : "";
  return `${sign}${format(value)}`;
}

function describeAttempt(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "bootstrap";
  if (Math.abs(Number(value)) < 1e-9) return "tied parent, not kept";
  return signed(value);
}

function deltaClass(value) {
  if (value === null || value === undefined) return "delta neutral";
  if (value > 0) return "delta positive";
  if (value > -0.02) return "delta near";
  return "delta negative";
}

function lossDeltaClass(value) {
  if (value === null || value === undefined) return "delta neutral";
  if (value < 0) return "delta positive";
  if (value < 0.02) return "delta near";
  return "delta negative";
}

function summarizeLoss(events) {
  if (!events.length) return { bestLoss: null, worstLoss: null };
  return {
    bestLoss: Math.min(...events.map((event) => event.validationLoss ?? 0)),
    worstLoss: Math.max(...events.map((event) => event.validationLoss ?? 0)),
  };
}

function summarizeChanges(events) {
  const groups = new Map();
  for (const event of events) {
    const label = shortHypothesis(event);
    const key = label.split(" ").slice(0, 3).join(" ");
    const group = groups.get(key) ?? { key, label, events: [] };
    group.events.push(event);
    groups.set(key, group);
  }
  return [...groups.values()]
    .map((group) => {
      const sorted = [...group.events].sort((a, b) => (a.validationLoss ?? 0) - (b.validationLoss ?? 0));
      const averageLoss = group.events.reduce((sum, event) => sum + (event.validationLoss ?? 0), 0) / group.events.length;
      return {
        key: group.key,
        label: group.label,
        count: group.events.length,
        best: sorted[0],
        averageLoss,
      };
    })
    .sort((a, b) => (a.best.validationLoss ?? 0) - (b.best.validationLoss ?? 0));
}

createRoot(document.getElementById("root")).render(<App />);
