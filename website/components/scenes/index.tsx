"use client";

import type { L } from "@/lib/i18n";
import { answerScene, chatRun, listeningScene, modesScene, permissionScene, statusScene, toolsScene } from "./chat";
import { contextScene, gitScene, memoryScene, planScene } from "./panels";
import { dashboardScene, gpuScene, hardwareScene, liveScene, readyScene, requestsScene, thinkingScene } from "./monitor";
import { legacyChat, legacyMonitor } from "./legacy";
import { marksScene, permissionsScene, projectScene } from "./settings";
import { Scene, type SceneDef } from "./Scene";

/** Every scene the site plays. A scene is a piece of the app's interface, rebuilt here and played from a clock. */
export const SCENES = {
  "chat-run": chatRun,
  listening: listeningScene,
  thinking: thinkingScene,
  ready: readyScene,
  tools: toolsScene,
  status: statusScene,
  gpus: gpuScene,
  plan: planScene,
  answer: answerScene,
  permission: permissionScene,
  modes: modesScene,
  git: gitScene,
  memory: memoryScene,
  context: contextScene,
  dashboard: dashboardScene,
  live: liveScene,
  requests: requestsScene,
  hardware: hardwareScene,
  permissions: permissionsScene,
  project: projectScene,
  marks: marksScene,
  "legacy-chat": legacyChat,
  "legacy-monitor": legacyMonitor,
} satisfies Record<string, SceneDef>;

export type SceneId = keyof typeof SCENES;

export function AppScene({ id, alt, active = true, className = "", replay = true, lang, glide = false }: { id: SceneId; alt: L; active?: boolean; className?: string; replay?: boolean; lang?: "en" | "th"; glide?: boolean }) {
  return <Scene def={SCENES[id]} alt={alt} active={active} className={className} replay={replay} lang={lang} glide={glide} />;
}
