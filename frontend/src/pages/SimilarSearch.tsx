import { Search } from "lucide-react";

const mockProducts = [
  { id: "A001", title: "Organic almond butter 16 oz", price: "$12.49", similarity: "0.92" },
  { id: "A002", title: "Natural peanut butter jar", price: "$8.99", similarity: "0.87" },
  { id: "A003", title: "Cashew spread unsweetened", price: "$14.25", similarity: "0.83" },
];

export default function SimilarSearch() {
  return (
    <div className="page-shell">
      <div className="mb-6">
        <h1 className="text-3xl font-semibold tracking-normal text-ink">Similar Product Search</h1>
        <p className="mt-2 text-sm text-slate-600">FAISS-backed retrieval placeholder with mock product cards.</p>
      </div>
      <section className="panel p-6">
        <label className="field-label" htmlFor="similar-query">
          Text query
        </label>
        <div className="mt-2 flex flex-col gap-3 sm:flex-row">
          <input
            id="similar-query"
            className="field-input"
            placeholder="Search by product title, attributes, or category"
          />
          <button className="inline-flex items-center justify-center gap-2 rounded-md bg-steel px-4 py-2 text-sm font-semibold text-white hover:bg-steel/90">
            <Search className="h-4 w-4" aria-hidden="true" />
            Search
          </button>
        </div>
      </section>
      <section className="mt-6 grid gap-4 md:grid-cols-3">
        {mockProducts.map((product) => (
          <article key={product.id} className="panel p-5">
            <p className="text-xs font-medium text-slate-500">{product.id}</p>
            <h2 className="mt-2 text-base font-semibold text-ink">{product.title}</h2>
            <p className="mt-3 text-sm text-slate-600">Price: {product.price}</p>
            <p className="mt-1 text-sm text-slate-600">Similarity: {product.similarity}</p>
          </article>
        ))}
      </section>
    </div>
  );
}
