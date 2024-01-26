import { pipeline, cos_sim } from "@xenova/transformers";

export async function embed() {
  // Define the query and sentences
  const query = "I work at Kin";
  const sentences = [
    "thomas works at Kin",
    "kenji works at Kin",
    "Christopher works at-> Kin",
    "Me co-founded Kin",
    "Andrei works at Kin",
    "Me is cto of Kin",
    "Volodymyr works at Kin",
    "Me reached at kin feature freeze",
    "Me vents to Kin",
    "0.1.1 version of Kin",
    "Kin offers help Me",
    "0.1.4 version of Kin",
    "Kin requires responsibility",
    "Me looks for in candidates align with Kin's values",
    "Kin values transparency and trustworthiness",
    "Me works at Kin",
    "I work at Kin",
    "Me works at Kin",
  ];

  // Load the pipeline for embeddings
  const extractor = await pipeline(
    "feature-extraction",
    "Xenova/all-MiniLM-L6-v2",
    { quantized: false } // Comment out this line to use the quantized version
  );

  // Compute embeddings for the query and all sentences
  const queryEmbedding = await extractor([query], {
    pooling: "mean",
    normalize: true,
  });

  const sentenceEmbeddings = await Promise.all(
    sentences.map((sentence) =>
      extractor([sentence], { pooling: "mean", normalize: true })
    )
  );

  // Compute cosine similarities
  const cosineScores = sentenceEmbeddings.map((sentenceEmbedding) =>
    cos_sim(queryEmbedding[0].data, sentenceEmbedding[0].data)
  );

  // Combine sentences with their scores and sort by similarity
  const sentenceScorePairs = sentences.map((sentence, index) => ({
    sentence: sentence,
    score: cosineScores[index],
  }));

  const sortedSentences = sentenceScorePairs.sort((a, b) => b.score - a.score);

  // Output the sorted sentences with their similarity scores
  sortedSentences.forEach((pair) => {
    console.log(`${pair.sentence}, ${pair.score}`);
  });
}
