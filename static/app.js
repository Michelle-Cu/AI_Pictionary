// Shared utilities — page-specific logic lives in each template.

function getLocalIP() {
  return window.location.hostname;
}

function buildGroupURL(team, group, token, base) {
  base = base || window.location.origin;
  return `${base}/team/${team}/group/${group}?token=${token}`;
}
