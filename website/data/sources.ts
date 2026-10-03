/** The revisions every statement on the site is based on (checked 2026-10-03). */
export const REV = {
  fork: { repo: "xenodeve/Strata-xeno", sha: "73649e0", full: "73649e0", date: "2026-10-03", note: "PR #120" },
  upstream: { repo: "Niko1221/Strata", sha: "99f3dbd", date: "2026-10-03", note: "v0.1.38" },
  base: { sha: "db4f91a", note: "upstream v0.1.37" },
} as const;

const GH = "https://github.com";

/** A link to a file in the fork at the revision the site was written against. */
export const file = (path: string) => `${GH}/${REV.fork.repo}/blob/${REV.fork.sha}/${path}`;
export const issue = (n: number) => `${GH}/${REV.fork.repo}/issues/${n}`;
export const pr = (n: number) => `${GH}/${REV.fork.repo}/pull/${n}`;

export const LINKS = {
  fork: `${GH}/${REV.fork.repo}`,
  upstream: `${GH}/${REV.upstream.repo}`,
  upstreamLicense: `${GH}/${REV.upstream.repo}/blob/main/LICENSE`,
  forkLicense: `${GH}/${REV.fork.repo}/blob/main/LICENSE`,
  uiReferences: file("serve/ui/REFERENCES.md"),
  agents: file("AGENTS.md"),
  blueprint: file("docs/BLUEPRINT.md"),
} as const;
