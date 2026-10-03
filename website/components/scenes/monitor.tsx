"use client";

import type { SceneBodyProps, SceneDef } from "./Scene";
import { DashboardBody, LIVE_END, LiveBody, ReadyBody } from "./monitor-a-parts";
import { Experts, GPUS_DURATION, GpusPage, HARDWARE_DURATION, Hardware, HardwarePage, REQUESTS_DURATION, RequestRows, RequestsPage } from "./monitor-b-parts";

/**
 * The monitor pages of the app, as scenes. The figures are the ones one real run showed (the same run as the chat scenes):
 * they say what that screen showed, on that PC, once. They are not a benchmark. The Thai strings are the app's own.
 */

/** Mood "searching": the dashboard at rest, "Ready", with the speed of the last four requests. */
export const readyScene: SceneDef = {
  // the page sharpens in (260 ms) and its head and block rise 0 and 50 ms after it (420 ms): the app moves nothing else on it
  // (components/scenes/monitor-a-parts.tsx)
  duration: 500,
  Body: ({ ms, lang }: SceneBodyProps) => <ReadyBody ms={ms} lang={lang} />,
};

/** The whole dashboard: the state, the speed, the requests, the hardware, where the experts ran. */
export const dashboardScene: SceneDef = {
  // everything has landed by 650 ms; the rest is the window following the page down when it is cut short (it is a no-op when the page fits)
  duration: 2300,
  Body: ({ ms, lang }: SceneBodyProps) => <DashboardBody ms={ms} lang={lang} />,
};

/** The Live page while the model thinks: a count of tokens, the speed, the line of the last minute, the cards, the recent requests. */
export const liveScene: SceneDef = {
  // the page is read every 500 ms while a request runs: the count of tokens and the elapsed time pop in at each reading, the chart gets a point as the page reads
  duration: LIVE_END + 300,
  Body: ({ ms, lang }: SceneBodyProps) => <LiveBody ms={ms} lang={lang} full />,
};

/** The thinking mood on its own: "Thinking · N tokens" with the speed and elapsed time. */
export const thinkingScene: SceneDef = {
  duration: LIVE_END + 300,
  Body: ({ ms, lang }: SceneBodyProps) => <LiveBody ms={ms} lang={lang} full={false} />,
};

/** The Requests page: every request, with what was read, cached and written, and the speeds. */
export const requestsScene: SceneDef = {
  // the page sharpens in and its children rise 50 ms apart; the rows come in with it (components/scenes/monitor-b-parts.tsx)
  duration: REQUESTS_DURATION,
  Body: ({ ms, lang }: SceneBodyProps) => <RequestsPage ms={ms} lang={lang} />,
};

/** The Hardware page: the GPUs, the CPU and the expert kernel it uses, the memory. */
export const hardwareScene: SceneDef = {
  // the page sharpens in and its children rise 50 ms apart; the figures and bars are there (the app does not count them up)
  duration: HARDWARE_DURATION,
  Body: ({ ms, lang }: SceneBodyProps) => <HardwarePage ms={ms} lang={lang} />,
};


/** Mood "working": the GPUs and the CPU, shown plainly (the middle of the Hardware page). */
export const gpuScene: SceneDef = {
  // the two sections are the third and fourth children of the page: they rise 100 and 150 ms after it sharpens in
  duration: GPUS_DURATION,
  Body: ({ ms, lang }: SceneBodyProps) => <GpusPage ms={ms} lang={lang} />,
};
