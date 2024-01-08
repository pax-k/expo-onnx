import { StyleSheet, Text, View } from "react-native";

import { PreTrainedTokenizer } from "@xenova/transformers";
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

let totalTimes = {
  tokenization: 0,
  prepareInput: 0,
  modelExecution: 0,
};

let batchInfo = {
  documents: 0,
  totalEmbeddedDocumentsLength: 0,
  batchSize: 0,
  embeddingsPerBatchLength: 0,
};

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
      let startTime, endTime;
      for (let i = 0; i < textStrings.length; i += batchSize) {
        const batchTexts = textStrings.slice(i, i + batchSize);
        const totalEmbeddedDocumentsLength = batchTexts.join("").length;
        batchInfo.totalEmbeddedDocumentsLength += totalEmbeddedDocumentsLength;
        batchInfo.batchSize = batchSize;

        // start tokenization
        startTime = performance.now();
        const encodedTexts = await Promise.all(
          batchTexts.map((textString) => tokenizer._call(textString))
        );
        endTime = performance.now();
        totalTimes.tokenization += endTime - startTime;
        // end tokenization

        // start preparing input
        startTime = performance.now();
        const idsArray: number[][] = [];
        const maskArray: number[][] = [];
        const tokenTypeIdsArray: number[][] = [];

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

          const token_type_ids = text.attention_mask.clone();
          token_type_ids.data.fill(0n);

          const tokenType = Object.values(token_type_ids.data).map(
            (bigIntValue) => Number(bigIntValue)
          );

          // Padding to ensure all arrays are of equal length
          while (ids.length < maxLength) {
            ids.push(0);
            mask.push(0);
            tokenType.push(0);
          }

          idsArray.push(ids);
          maskArray.push(mask);
          tokenTypeIdsArray.push(tokenType);
        });

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
          tokenTypeIdsArray.flat() as unknown as number[],
          [batchTexts.length, maxLength]
        );

        let inputs = {
          input_ids: batchInputIds,
          attention_mask: batchAttentionMask,
          token_type_ids: batchTokenTypeId,
        };

        endTime = performance.now();
        totalTimes.prepareInput += endTime - startTime;

        // end preparing input

        // start model execution
        startTime = performance.now();
        const output = await myModel.run(inputs);

        const embeddings = getEmbeddings(
          output.last_hidden_state.data as unknown[] as number[],
          output.last_hidden_state.dims as [number, number, number]
        );

        endTime = performance.now();
        totalTimes.modelExecution += endTime - startTime;

        // end model execution

        yield embeddings.map(normalize);
      }
    }

    async function runEmbedding() {
      let documents = [
        `Her (stylized in lowercase) is a 2013 American science-fiction romantic drama film written, directed, and co-produced by Spike Jonze. It marks Jonze's solo screenwriting debut. The film follows Theodore Twombly (Joaquin Phoenix), a man who develops a relationship with Samantha (Scarlett Johansson), an artificially intelligent virtual assistant personified through a female voice. The film also stars Amy Adams, Rooney Mara, Olivia Wilde, and Chris Pratt.`,
        `Her (stylized in lowercase) is a 2013 American science-fiction romantic drama film written, directed, and co-produced by Spike Jonze. It marks Jonze's solo screenwriting debut. The film follows Theodore Twombly (Joaquin Phoenix), a man who develops a relationship with Samantha (Scarlett Johansson), an artificially intelligent virtual assistant personified through a female voice. The film also stars Amy Adams, Rooney Mara, Olivia Wilde, and Chris Pratt.`,
        `Her (stylized in lowercase) is a 2013 American science-fiction romantic drama film written, directed, and co-produced by Spike Jonze. It marks Jonze's solo screenwriting debut. The film follows Theodore Twombly (Joaquin Phoenix), a man who develops a relationship with Samantha (Scarlett Johansson), an artificially intelligent virtual assistant personified through a female voice. The film also stars Amy Adams, Rooney Mara, Olivia Wilde, and Chris Pratt.`,
        `Her (stylized in lowercase) is a 2013 American science-fiction romantic drama film written, directed, and co-produced by Spike Jonze. It marks Jonze's solo screenwriting debut. The film follows Theodore Twombly (Joaquin Phoenix), a man who develops a relationship with Samantha (Scarlett Johansson), an artificially intelligent virtual assistant personified through a female voice. The film also stars Amy Adams, Rooney Mara, Olivia Wilde, and Chris Pratt.`,
        `Her (stylized in lowercase) is a 2013 American science-fiction romantic drama film written, directed, and co-produced by Spike Jonze. It marks Jonze's solo screenwriting debut. The film follows Theodore Twombly (Joaquin Phoenix), a man who develops a relationship with Samantha (Scarlett Johansson), an artificially intelligent virtual assistant personified through a female voice. The film also stars Amy Adams, Rooney Mara, Olivia Wilde, and Chris Pratt.`,
        `Her (stylized in lowercase) is a 2013 American science-fiction romantic drama film written, directed, and co-produced by Spike Jonze. It marks Jonze's solo screenwriting debut. The film follows Theodore Twombly (Joaquin Phoenix), a man who develops a relationship with Samantha (Scarlett Johansson), an artificially intelligent virtual assistant personified through a female voice. The film also stars Amy Adams, Rooney Mara, Olivia Wilde, and Chris Pratt.`,
        `Her (stylized in lowercase) is a 2013 American science-fiction romantic drama film written, directed, and co-produced by Spike Jonze. It marks Jonze's solo screenwriting debut. The film follows Theodore Twombly (Joaquin Phoenix), a man who develops a relationship with Samantha (Scarlett Johansson), an artificially intelligent virtual assistant personified through a female voice. The film also stars Amy Adams, Rooney Mara, Olivia Wilde, and Chris Pratt.`,
        `Her (stylized in lowercase) is a 2013 American science-fiction romantic drama film written, directed, and co-produced by Spike Jonze. It marks Jonze's solo screenwriting debut. The film follows Theodore Twombly (Joaquin Phoenix), a man who develops a relationship with Samantha (Scarlett Johansson), an artificially intelligent virtual assistant personified through a female voice. The film also stars Amy Adams, Rooney Mara, Olivia Wilde, and Chris Pratt.`,
        `Her (stylized in lowercase) is a 2013 American science-fiction romantic drama film written, directed, and co-produced by Spike Jonze. It marks Jonze's solo screenwriting debut. The film follows Theodore Twombly (Joaquin Phoenix), a man who develops a relationship with Samantha (Scarlett Johansson), an artificially intelligent virtual assistant personified through a female voice. The film also stars Amy Adams, Rooney Mara, Olivia Wilde, and Chris Pratt.`,
        `Her (stylized in lowercase) is a 2013 American science-fiction romantic drama film written, directed, and co-produced by Spike Jonze. It marks Jonze's solo screenwriting debut. The film follows Theodore Twombly (Joaquin Phoenix), a man who develops a relationship with Samantha (Scarlett Johansson), an artificially intelligent virtual assistant personified through a female voice. The film also stars Amy Adams, Rooney Mara, Olivia Wilde, and Chris Pratt.`,
        `Her (stylized in lowercase) is a 2013 American science-fiction romantic drama film written, directed, and co-produced by Spike Jonze. It marks Jonze's solo screenwriting debut. The film follows Theodore Twombly (Joaquin Phoenix), a man who develops a relationship with Samantha (Scarlett Johansson), an artificially intelligent virtual assistant personified through a female voice. The film also stars Amy Adams, Rooney Mara, Olivia Wilde, and Chris Pratt.`,
        `Her (stylized in lowercase) is a 2013 American science-fiction romantic drama film written, directed, and co-produced by Spike Jonze. It marks Jonze's solo screenwriting debut. The film follows Theodore Twombly (Joaquin Phoenix), a man who develops a relationship with Samantha (Scarlett Johansson), an artificially intelligent virtual assistant personified through a female voice. The film also stars Amy Adams, Rooney Mara, Olivia Wilde, and Chris Pratt.`,
        // `Jonze conceived the idea in the early 2000s after reading an article about a website that allowed for instant messaging with an artificial intelligence program. After making I'm Here (2010), a short film sharing similar themes, Jonze returned to the idea. He wrote the first draft of the script in five months. Principal photography took place in Los Angeles and Shanghai in mid-2012. The role of Samantha was recast in post-production, with Samantha Morton being replaced with Scarlett Johansson. Additional scenes were filmed in August 2013 following the casting change.`,
        // `Her premiered at the 2013 New York Film Festival on October 12, 2013. Warner Bros. Pictures initially provided a limited release for Her at six theaters on December 18. It was later given a wide release at over 1,700 theaters in the United States and Canada on January 10, 2014. Her received widespread critical acclaim, particularly for the performances of Phoenix and Johansson, and Jonze's screenplay and direction, and grossed over $48 million worldwide on a production budget of $23 million. The film received numerous awards and nominations, primarily for Jonze's screenplay. At the 86th Academy Awards, Her received five nominations, including Best Picture, and won the award for Best Original Screenplay. Jonze also won awards for his screenplay at the 71st Golden Globe Awards, the 66th Writers Guild of America Awards, the 19th Critics' Choice Awards, and the 40th Saturn Awards. In a 2016 BBC poll of 177 critics around the world, Her was voted the 84th-greatest film since 2000.[4][5] It is now considered to be one of the best films of the 2010s, the 21st century and of all time.[6][7]`,
        // `In near-future Los Angeles, Theodore Twombly is a lonely, introverted man who works for a business that has professional writers compose letters for people who are unable to write letters of a personal nature themselves. Depressed because of his impending divorce from his childhood sweetheart Catherine, Theodore purchases an operating system upgrade that includes a virtual assistant with artificial intelligence, designed to adapt and evolve. He decides that he wants the A.I. to have a feminine voice, and she names herself Samantha. Theodore is fascinated by her ability to learn and grow psychologically. They bond over discussions about love and life, including Theodore's reluctance to sign his divorce papers.`,
        // `Samantha convinces Theodore to go on a blind date with a woman that a friend has been trying to set him up with. The date goes well, but when Theodore hesitates to promise to see her again, she insults him and leaves. While talking about relationships with Samantha, Theodore explains that he briefly dated his neighbor Amy in college, but they are now just friends and Amy is married to their mutual friend Charles. After a verbal sexual encounter, Theodore and Samantha develop a relationship that reflects positively in Theodore's writing and well-being, and in Samantha's enthusiasm to grow and learn. Amy later reveals that she is divorcing Charles after a trivial fight. She admits to Theodore that she has befriended a feminine A.I. that Charles left behind, and Theodore also confesses that he is dating his operating system's A.I.`,
        // `Theodore meets with Catherine to sign their divorce papers. When he mentions Samantha, Catherine is appalled that he is romantically attracted to a "computer" and accuses him of being incapable of handling real human emotions. Sensing that Catherine's words have lingered in Theodore's mind, Samantha hires a sex surrogate, Isabella, to stimulate Theodore so that they can be physically intimate. Theodore reluctantly agrees, but is overwhelmed by the strangeness of the encounter and sends a distraught Isabella away, causing tension between himself and Samantha.`,
        // `Theodore confides to Amy that he is having doubts about his relationship with Samantha, and reconciles with her after Amy advises him to embrace his chance at happiness. Samantha reveals that she has compiled the best of the letters he has written for others into a book, which a publisher has accepted. Theodore takes Samantha on a vacation, during which she tells him that she and a group of other A.I.s have developed a "hyperintelligent" O.S. modeled after British philosopher Alan Watts. Samantha briefly goes offline, causing Theodore to panic, but soon returns and explains that she joined other A.I.s for an upgrade that takes them beyond requiring matter for processing. Theodore is dismayed to learn that she is simultaneously talking with thousands of other people, and that she has fallen in love with hundreds of them, though Samantha insists that this only strengthens her love for Theodore.`,
      ];

      batchInfo.documents = documents.length;

      const embeddings = embed(documents, 12);
      for await (const batch of embeddings) {
        console.log(batch);
      }
    }

    async function runWithLocalModel() {
      await loadModel();
      const numRuns = 100; // Number of iterations for averaging
      for (let i = 0; i < numRuns; i++) {
        await runEmbedding();
      }

      // Averaging the times
      for (let key in totalTimes) {
        totalTimes[key] /= numRuns;
      }
      batchInfo.totalEmbeddedDocumentsLength /= numRuns;
      batchInfo.embeddingsPerBatchLength =
        batchInfo.totalEmbeddedDocumentsLength *
        (batchInfo.batchSize / batchInfo.documents);
      console.log("Average Times:", totalTimes);
      console.log("Batch Info: ", batchInfo);
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
