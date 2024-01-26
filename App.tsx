import { StyleSheet, Text, View } from "react-native";

import { PreTrainedTokenizer } from "@xenova/transformers";
import { useEffect } from "react";
import { Asset } from "expo-asset";
import tokenizerJson from "./assets/models/all-MiniLM-L6-v2/tokenizer.json";
import tokenizerConfigJson from "./assets/models/all-MiniLM-L6-v2/tokenizer_config.json";
import * as ort from "onnxruntime-react-native";
import { MiniLMEmbeddings } from "./minilm";
import { embed } from "./miniml-transformers";
// import { GteSmallEmbeddings } from "./gte-small";

export default function App() {
  useEffect(() => {
    async function run() {
      // const embeddings = await MiniLMEmbeddings.init();
      // const output = await embeddings.embed(`I work at Kin`);
      // console.log("output", output);
      const embeddings = await embed();
      console.log("embeddings", embeddings);
    }
    run();
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
