const sections = [
  {
    title: "Architecture",
    text: "Python ML pipeline, FastAPI inference backend, React dashboard, and optional FAISS retrieval.",
  },
  {
    title: "Dataset",
    text: "Amazon ML Challenge train.csv and test.csv live under data/raw, with downloaded images under data/images.",
  },
  {
    title: "Limitations",
    text: "The first model is a text baseline. External marketplace prices are not used for training.",
  },
];

export default function About() {
  return (
    <div className="page-shell">
      <div className="mb-6">
        <h1 className="text-3xl font-semibold tracking-normal text-ink">About PriceLens</h1>
        <p className="mt-2 max-w-3xl text-sm text-slate-600">
          A resume-grade ML full-stack project with a strict boundary between training data and future market-comparison features.
        </p>
      </div>
      <section className="grid gap-4 lg:grid-cols-3">
        {sections.map((section) => (
          <article key={section.title} className="panel p-5">
            <h2 className="text-base font-semibold text-ink">{section.title}</h2>
            <p className="mt-3 text-sm leading-6 text-slate-600">{section.text}</p>
          </article>
        ))}
      </section>
    </div>
  );
}
