import { InferenceSession, TypedTensor } from "onnxruntime-react-native";
import { BertTokenizer } from "@xenova/transformers";
import { Asset } from "expo-asset";
import tokenizerJson from "./assets/models/all-MiniLM-L6-v2/tokenizer.json";
import tokenizerConfigJson from "./assets/models/all-MiniLM-L6-v2/tokenizer_config.json";

function cosineSimilarity(vecA: number[], vecB: number[]) {
  let dotProduct = 0.0;
  let normA = 0.0;
  let normB = 0.0;
  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

function euclideanDistance(vecA, vecB) {
  let sum = 0;

  for (let i = 0; i < vecA.length; i++) {
    sum += (vecA[i] - vecB[i]) ** 2;
  }

  return Math.sqrt(sum);
}

function manhattanDistance(vecA, vecB) {
  let sum = 0;

  for (let i = 0; i < vecA.length; i++) {
    sum += Math.abs(vecA[i] - vecB[i]);
  }

  return sum;
}

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

  // Load the ONNX model
  const modelAsset = await Asset.loadAsync(
    require("./assets/models/all-MiniLM-L6-v2/onnx/model.onnx")
  );
  const { localUri } = modelAsset[0];
  const session = await InferenceSession.create(localUri);

  // const tokenizer = new PreTrainedTokenizer(tokenizerJson, tokenizerConfigJson);
  const tokenizer = new BertTokenizer(tokenizerJson, tokenizerConfigJson);

  const prepareInput = (text) => {
    const model_inputs = tokenizer._call(text, {
      padding: true,
      truncation: true,
    });

    return model_inputs;
  };

  const queryInput = prepareInput(query);
  const sentenceInputs = sentences.map(prepareInput);

  // Function to extract embeddings
  const extractEmbedding = async (input: {
    input_ids: TypedTensor<"int64">;
    attention_mask: TypedTensor<"int64">;
    token_type_ids: TypedTensor<"int64">;
  }) => {
    let output = await session.run(input);
    // output = mean_pooling(output.last_hidden_state, input.attention_mask);
    // output = output.normalize(2, -1);
    // @ts-ignore
    return Array.from(output.last_hidden_state.data);
  };

  // Compute embeddings
  const queryEmbedding = await extractEmbedding(queryInput);

  const sentenceEmbeddings = await Promise.all(
    sentenceInputs.map(extractEmbedding)
  );

  // Compute cosine similarities
  const cosineScores = sentenceEmbeddings.map((sentenceEmbedding) =>
    // @ts-ignore
    cosineSimilarity(queryEmbedding, sentenceEmbedding)
  );

  const euclideanDistanceScores = sentenceEmbeddings.map((sentenceEmbedding) =>
    // @ts-ignore
    euclideanDistance(queryEmbedding, sentenceEmbedding)
  );

  const manhattanDistanceScores = sentenceEmbeddings.map((sentenceEmbedding) =>
    // @ts-ignore
    manhattanDistance(queryEmbedding, sentenceEmbedding)
  );

  // Combine sentences with their scores and sort by similarity
  const sentenceScorePairs = sentences.map((sentence, index) => {
    console.log(sentenceInputs[index].input_ids);
    return {
      sentence: sentence,
      cosineScore: cosineScores[index],
      euclideanDistanceScores: euclideanDistanceScores[index],
      manhattanDistanceScores: manhattanDistanceScores[index],
      embeddings: sentenceEmbeddings[index],
    };
  });

  const sortedSentences = sentenceScorePairs.sort(
    (a, b) => b.cosineScore - a.cosineScore
  );

  // Output the sorted sentences with their similarity scores
  sortedSentences.forEach((pair) => {
    console.log(`${pair.sentence}, ${pair.cosineScore}`);
  });

  console.log("sortedSentences", JSON.stringify(sortedSentences));
}
