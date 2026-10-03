import type { Metadata } from "next";
import { DetailsApp } from "@/components/details/DetailsApp";
import { Footer } from "@/components/Footer";
import { Waves } from "@/components/home/Waves";
import { Nav } from "@/components/Nav";

export const metadata: Metadata = {
  title: "Details: Strata-xeno",
  description: "Everything Strata-xeno adds, topic by topic: the app, your setup, what you can see, the API, the engine and the numbers.",
};

export default function DetailsPage() {
  return (
    <>
      <Nav />
      <main id="main">
        <DetailsApp />
        <Waves />
      </main>
      <Footer />
    </>
  );
}
