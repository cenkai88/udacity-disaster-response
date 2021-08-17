<template>
  <div>
    <Header />
    <div class="row">
      <Search />
      <Pie :pieData="genre_distribution" />
    </div>
    <div class="row">
      <Line :lineData="category_count" />
    </div>
  </div>
</template>

<script>
const axios = require("axios");

import Header from "./components/Header.vue";
import Search from "./components/Search.vue";
import Pie from "./components/Pie.vue";
import Line from "./components/Line.vue";

export default {
  name: "App",
  components: {
    Header,
    Search,
    Pie,
    Line,
  },
  data() {
    return {
      genre_distribution: [],
      category_count: [],
    };
  },
  async mounted() {
    const { data } = await axios.post(
      `http://localhost:3000/message/message_stat/v1`
    );
    this.genre_distribution = data.genre_distribution;
    this.category_count = data.category_count;
  },
};
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
}
body {
  margin: 0;
}
.row {
  display: flex;
  justify-content: space-evenly;
  margin: 24px 0;
}
</style>
