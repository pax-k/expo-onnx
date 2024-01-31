import { StyleSheet, Text, View } from "react-native";

import { useEffect } from "react";
import { embed } from "./minilm";

export default function App() {
  useEffect(() => {
    async function run() {
      function testFloatingPointPrecisionAdvanced() {
        // Dot product of large vectors
        const vectorSize = 10000;
        const vector1 = Array.from({ length: vectorSize }, (_, i) =>
          Math.random()
        );
        const vector2 = Array.from({ length: vectorSize }, (_, i) =>
          Math.random()
        );
        let dotProduct = 0;
        for (let i = 0; i < vectorSize; i++) {
          dotProduct += vector1[i] * vector2[i];
        }

        // BigInt operation
        const largeInt = BigInt(1e16);
        const anotherLargeInt = BigInt(1e16 - 1);
        const bigIntProduct = largeInt * anotherLargeInt;

        // Operations with floating points and BigInts
        const mixedOperation =
          parseFloat(bigIntProduct.toString()) / dotProduct;

        // Repeated multiplications with small floating point numbers
        let repeatedMultiplication = 0.0001;
        for (let i = 0; i < 10000; i++) {
          repeatedMultiplication *= 1.0001;
        }

        return {
          dotProduct,
          bigIntProduct: bigIntProduct.toString(), // Convert BigInt to string for logging
          mixedOperation,
          repeatedMultiplication,
        };
      }

      // Example usage
      const precisionResultsAdvanced = testFloatingPointPrecisionAdvanced();

      await embed();
      console.log(precisionResultsAdvanced);
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
