// module.exports = function(api) {
//   api.cache(true);
//   return {
//     presets: ['babel-preset-expo'],
//   };
// };

module.exports = {
  // presets: ['module:metro-react-native-babel-preset'],
  presets: ["module:metro-react-native-babel-preset", "babel-preset-expo"],
  plugins: [
    "@babel/plugin-proposal-export-namespace-from",
    "babel-plugin-transform-import-meta",
    [
      "module-resolver",
      {
        alias: {
          buffer: "@craftzdog/react-native-buffer",
        },
      },
    ],
  ],
};
