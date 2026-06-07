import { BarChart3, BrainCircuit, Search, ShoppingBasket } from "lucide-react";

const highlights = [
  { title: "Multimodal ML", detail: "Catalog text now, image features staged next.", icon: BrainCircuit },
  { title: "Price prediction", detail: "FastAPI inference contract connected to React.", icon: ShoppingBasket },
  { title: "Model evaluation", detail: "SMAPE, MAE, RMSE, and R2 tracked from baseline runs.", icon: BarChart3 },
  { title: "Similar search", detail: "FAISS retrieval planned after baseline artifacts stabilize.", icon: Search },
];

export default function Home() {
  return (
    <div className="page-shell">
      <section className="py-6">
        <h1 className="max-w-3xl text-4xl font-semibold tracking-normal text-ink">
          Production-style product price intelligence for e-commerce catalogs.
        </h1>
        <p className="mt-4 max-w-3xl text-base text-slate-600">
          PriceLens predicts product prices from catalog content, images, and metadata using the
          Amazon ML Challenge dataset. External marketplace prices stay out of model training.
        </p>
      </section>
      <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        {highlights.map((item) => (
          <article key={item.title} className="panel p-5">
            <item.icon className="h-6 w-6 text-steel" aria-hidden="true" />
            <h2 className="mt-4 text-base font-semibold text-ink">{item.title}</h2>
            <p className="mt-2 text-sm text-slate-600">{item.detail}</p>
          </article>
        ))}
      </section>
    </div>
  );
}
