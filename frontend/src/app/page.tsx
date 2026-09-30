export default function Home() {
  return (
    <main className="min-h-screen flex flex-col items-center justify-center p-8 bg-zinc-50 dark:bg-zinc-950 text-zinc-900 dark:text-zinc-100">
      <div className="max-w-2xl w-full text-center space-y-6">
        <div className="inline-block px-3 py-1 rounded-full text-xs font-semibold tracking-wide uppercase bg-blue-100 dark:bg-blue-950/60 text-blue-700 dark:text-blue-300 border border-blue-200 dark:border-blue-800">
          Municipal Complaint Intelligence System
        </div>
        <h1 className="text-4xl sm:text-5xl font-extrabold tracking-tight">
          Civic Issue Management & Intelligence Portal
        </h1>
        <p className="text-lg text-zinc-600 dark:text-zinc-400">
          AI-driven classification, geospatial duplicate detection, and automated priority routing for urban municipal grievances.
        </p>
        <div className="flex flex-wrap items-center justify-center gap-4 pt-4">
          <div className="flex items-center gap-2 px-4 py-2 rounded-lg bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 shadow-sm text-sm">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse"></span>
            <span>FastAPI Backend: Ready</span>
          </div>
          <div className="flex items-center gap-2 px-4 py-2 rounded-lg bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 shadow-sm text-sm">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse"></span>
            <span>PostGIS Database: Connected</span>
          </div>
        </div>
      </div>
    </main>
  );
}
