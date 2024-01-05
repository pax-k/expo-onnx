import { StyleSheet, Text, View } from "react-native";

import {
  pipeline,
  env,
  PreTrainedTokenizer,
  add_token_types,
} from "@xenova/transformers";
import { useEffect } from "react";
import { Asset } from "expo-asset";
import tokenizerJson from "./assets/models/all-MiniLM-L6-v2/tokenizer.json";
import tokenizerConfigJson from "./assets/models/all-MiniLM-L6-v2/tokenizer_config.json";
import * as ort from "onnxruntime-react-native";

function getEmbeddings(
  data: number[],
  dimensions: [number, number, number]
): number[][] {
  const [x, y, z] = dimensions;

  return Array.from({ length: x }, (_, index) => {
    const startIndex = index * y * z;
    const endIndex = startIndex + z;
    return data.slice(startIndex, endIndex);
  });
}

function normalize(v: number[]): number[] {
  const norm = Math.sqrt(v.reduce((acc, val) => acc + val * val, 0));
  const epsilon = 1e-12;

  return v.map((val) => val / Math.max(norm, epsilon));
}

export default function App() {
  useEffect(() => {
    let myModel;
    let tokenizer;
    async function loadModel() {
      const modelAsset = await Asset.loadAsync(
        require("./assets/models/all-MiniLM-L6-v2/onnx/model.onnx")
      );
      const modelUri = modelAsset[0].localUri;

      myModel = await ort.InferenceSession.create(modelUri);
      tokenizer = new PreTrainedTokenizer(tokenizerJson, tokenizerConfigJson);
    }

    async function* embed(textStrings: string[], batchSize: number = 256) {
      for (let i = 0; i < textStrings.length; i += batchSize) {
        const batchTexts = textStrings.slice(i, i + batchSize);

        const encodedTexts = await Promise.all(
          batchTexts.map((textString) => tokenizer._call(textString))
        );

        const idsArray: number[][] = [];
        const maskArray: number[][] = [];

        const maxLength = Math.max(
          ...encodedTexts.map((text) => text.input_ids.data.length)
        );

        encodedTexts.forEach((text) => {
          const ids = Object.values(text.input_ids.data).map((bigIntValue) =>
            Number(bigIntValue)
          );

          const mask = Object.values(text.attention_mask.data).map(
            (bigIntValue) => Number(bigIntValue)
          );

          // Padding to ensure all arrays are of equal length
          while (ids.length < maxLength) {
            ids.push(0); // Assuming 0 is the padding token ID
            mask.push(0); // Padding for attention mask
          }

          idsArray.push(ids);
          maskArray.push(mask);

          // const { token_type_ids } = add_token_types({ input_ids: idsArray });
          // typeIdsArray.push(token_type_ids);
        });

        const { token_type_ids } = add_token_types({ input_ids: idsArray });

        const batchInputIds = new ort.Tensor(
          "int64",
          idsArray.flat() as unknown as number[],
          [batchTexts.length, maxLength]
        );

        const batchAttentionMask = new ort.Tensor(
          "int64",
          maskArray.flat() as unknown as number[],
          [batchTexts.length, maxLength]
        );

        const batchTokenTypeId = new ort.Tensor(
          "int64",
          token_type_ids.flat() as unknown as number[],
          [batchTexts.length, maxLength]
        );

        let inputs = {
          input_ids: batchInputIds,
          attention_mask: batchAttentionMask,
          token_type_ids: batchTokenTypeId,
        };

        const output = await myModel.run(inputs);

        const embeddings = getEmbeddings(
          output.last_hidden_state.data as unknown[] as number[],
          output.last_hidden_state.dims as [number, number, number]
        );

        yield embeddings.map(normalize);
      }
    }

    async function runWithRemoteModel() {
      // Create a feature-extraction pipeline
      const extractor = await pipeline(
        "feature-extraction",
        "Xenova/all-MiniLM-L6-v2"
      );

      // Compute sentence embeddings
      const sentences = [
        "This is an example sentence",
        "Each sentence is converted",
      ];
      const output = await extractor(sentences, {
        pooling: "mean",
        normalize: true,
      });
      console.log(output);
    }

    async function runWithLocalModel() {
      await loadModel();
      let documents = [
        "passage: Hello, World!",
        "passage: Hello, World!!",
        "query: Hello, World!",
        "passage: This is an example passage.",
      ];

      const embeddings = embed(documents);
      for await (const batch of embeddings) {
        console.log(batch);
      }
    }

    runWithLocalModel();
  }, []);
  return (
    <View style={styles.container}>
      <Text>Open up App.tsx to start working on your app!</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#fff",
    alignItems: "center",
    justifyContent: "center",
  },
});
