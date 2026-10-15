// Client-side navigation: the app keeps its path in state and listens to popstate.
export function navigate(to: string) {
  window.history.pushState({}, '', to)
  window.dispatchEvent(new PopStateEvent('popstate'))
}

/** The one-page situation report for a job (rendered by the report page, printable as PDF). */
export const reportPath = (jobId: string) => `/report?job=${encodeURIComponent(jobId)}`
