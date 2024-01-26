/* eslint-disable import/namespace */

import * as ort from "onnxruntime-react-native";
import { PreTrainedTokenizer, Tensor } from "@xenova/transformers";
import { Asset } from "expo-asset";
import tokenizerJson from "./assets/models/gte-small/tokenizer.json";
import tokenizerConfigJson from "./assets/models/gte-small/tokenizer_config.json";

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

export class GteSmallEmbeddings {
  private model: ort.InferenceSession;
  private tokenizer: PreTrainedTokenizer;
  public static readonly modelName = "gte-small";

  private constructor(
    model: ort.InferenceSession,
    tokenizer: PreTrainedTokenizer
  ) {
    this.model = model;
    this.tokenizer = tokenizer;
  }

  public static async init() {
    try {
      const modelAsset = await Asset.loadAsync(
        require("./assets/models/gte-small/onnx/model.onnx")
      );
      const modelUri = modelAsset[0].localUri;
      if (!modelUri) {
        throw new Error("Model not found");
      }

      const model = await ort.InferenceSession.create(modelUri);
      const tokenizer = new PreTrainedTokenizer(
        tokenizerJson,
        tokenizerConfigJson
      );

      console.log("Model loaded successfully:", this.modelName);

      return new GteSmallEmbeddings(model, tokenizer);
    } catch (err) {
      console.error(`Failed to load model: ${err}`);
      throw err;
    }
  }

  async embed(textString: string): Promise<number[]> {
    try {
      let encodedText = this.tokenizer._call(textString, {
        truncation: false,
        // return_tensor: false,
      }) as Record<string, Tensor>;

      const ids = Object.values(encodedText.input_ids.data).map((bigIntValue) =>
        Number(bigIntValue)
      );

      const mask = Object.values(encodedText.attention_mask.data).map(
        (bigIntValue) => Number(bigIntValue)
      );

      const token_type_ids = encodedText.attention_mask.clone();
      token_type_ids.data.fill(0n);

      const tokenType = Object.values(token_type_ids.data).map((bigIntValue) =>
        Number(bigIntValue)
      );

      const maxLength = encodedText.input_ids.data.length;

      // Padding to ensure all arrays are of equal length
      while (ids.length < maxLength) {
        ids.push(0);
        mask.push(0);
        tokenType.push(0);
      }

      const batchInputIds = new ort.Tensor(
        "int64",
        ids.flat() as unknown as number[],
        [1, maxLength]
      );

      const batchAttentionMask = new ort.Tensor(
        "int64",
        mask.flat() as unknown as number[],
        [1, maxLength]
      );

      const batchTokenTypeId = new ort.Tensor(
        "int64",
        tokenType.flat() as unknown as number[],
        [1, maxLength]
      );

      const inputs = {
        input_ids: batchInputIds,
        attention_mask: batchAttentionMask,
        token_type_ids: batchTokenTypeId,
      };

      // end preparing input

      // start model execution
      const output = await this.model.run(inputs);

      const embeddings = getEmbeddings(
        output.last_hidden_state.data as unknown[] as number[],
        output.last_hidden_state.dims as [number, number, number]
      );

      // end model execution

      return normalize(embeddings[0]);
    } catch (err) {
      console.error(`Failed to embed text: ${err}`);
      throw err;
    }
  }

  // eslint-disable-next-line class-methods-use-this
  vectorOptions() {
    return {
      dimensions: 384,
      efConstruction: 16,
      maxConnections: 32,
    };
  }
}
