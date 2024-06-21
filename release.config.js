module.exports = {
  branches: ["main"],
  plugins: [
    [
      "@semantic-release/commit-analyzer",
      {
        preset: "angular",
        releaseRules: [
          { type: "build", release: "minor" },
          { type: "chore", release: "minor" },
          { type: "ci", release: false },
          { type: "docs", release: false },
          { type: "feat", release: "minor" },
          { type: "fix", release: "minor" },
          { type: "perf", release: "minor" },
          { type: "refactor", release: false },
          { type: "revert", release: "minor" },
          { type: "style", release: false },
          { type: "test", release: false },
        ]
      }
    ],
      "@semantic-release/release-notes-generator",
      "@semantic-release/changelog",
    [
      "@semantic-release/github",
      {
        successComment: false,
        failComment: false,
        failTitle: false
      }
    ]
  ]
};
