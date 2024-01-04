import { StyleSheet, Text, View } from "react-native";

import { pipeline } from "@xenova/transformers";
import { useEffect } from "react";

export default function App() {
  useEffect(() => {
    async function run() {
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
