import { Loader2, Send } from "lucide-react";
import { FormEvent, useState } from "react";

import type { PredictPayload } from "../types";
import ImageUpload from "./ImageUpload";

type ProductInputFormProps = {
  isSubmitting: boolean;
  onSubmit: (payload: PredictPayload) => Promise<void>;
};

export default function ProductInputForm({ isSubmitting, onSubmit }: ProductInputFormProps) {
  const [catalogContent, setCatalogContent] = useState("Organic almond butter 16 oz jar");
  const [brand, setBrand] = useState("Example Brand");
  const [category, setCategory] = useState("Grocery");
  const [image, setImage] = useState<File | null>(null);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    await onSubmit({
      catalog_content: catalogContent,
      brand,
      category,
      image,
    });
  }

  return (
    <form className="panel p-6" onSubmit={handleSubmit}>
      <div className="space-y-5">
        <div>
          <label className="field-label" htmlFor="catalog-content">
            Catalog content
          </label>
          <textarea
            id="catalog-content"
            className="field-input mt-2 min-h-40 resize-y"
            value={catalogContent}
            onChange={(event) => setCatalogContent(event.target.value)}
            required
          />
        </div>
        <div className="grid gap-4 sm:grid-cols-2">
          <div>
            <label className="field-label" htmlFor="brand">
              Brand
            </label>
            <input
              id="brand"
              className="field-input mt-2"
              value={brand}
              onChange={(event) => setBrand(event.target.value)}
              placeholder="Optional"
            />
          </div>
          <div>
            <label className="field-label" htmlFor="category">
              Category
            </label>
            <input
              id="category"
              className="field-input mt-2"
              value={category}
              onChange={(event) => setCategory(event.target.value)}
              placeholder="Optional"
            />
          </div>
        </div>
        <ImageUpload file={image} onChange={setImage} />
        <button
          type="submit"
          disabled={isSubmitting}
          className="inline-flex w-full items-center justify-center gap-2 rounded-md bg-mint px-4 py-3 text-sm font-semibold text-white transition hover:bg-mint/90 disabled:cursor-not-allowed disabled:opacity-70"
        >
          {isSubmitting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
          Generate prediction
        </button>
      </div>
    </form>
  );
}
