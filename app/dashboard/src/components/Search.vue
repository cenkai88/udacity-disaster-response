<template>
  <div class="search">
    <div class="title">Message Classification</div>
    <div>
      <div class="row">
        <input
          class="input"
          :value="inputValue"
          @input="updateInputValue"
          @change="updateInputValue"
          placeholder="Enter a message to classify"
        />
        <button @click="resetInputValue" class="button-light">reset</button>
        <button
          @click="sumbitValue"
          :disabled="inputValue.length == 0"
          class="button"
        >
          Submit
        </button>
      </div>
      <div>
        <div class="item-container" v-if="resultMap.size > 0">
          <div
            :class="['item', resultMap.get(key) ? 'item-selected' : '']"
            :key="key"
            v-for="key in resultMap.keys()"
          >
            {{ key }}
          </div>
        </div>
        <div v-if="resultMap.size === 0" class="item-placeholder">
          🔍 categories are empty, pls sumbit message first
        </div>
      </div>
    </div>
  </div>
</template>

<script>
const axios = require("axios");

export default {
  data() {
    return {
      loading: false,
      inputValue: "",
      resultMap: new Map(),
    };
  },
  methods: {
    updateInputValue({ target }) {
      this.inputValue = target.value;
    },
    resetInputValue() {
      this.inputValue = "";
      this.resultMap = new Map();
    },
    async sumbitValue() {
      this.loading = true;
      const { data } = await axios.post(
        `http://localhost:3000/message/message_classification/v1`,
        { text: this.inputValue }
      );
      this.resultMap = new Map(
        Object.keys(data).map((key) => [key, data[key] !== "0"])
      );
      this.loading = false;
    },
  },
};
</script>

<style scoped>
.search {
  width: calc(50vw - 32px);
  box-sizing: border-box;
  box-shadow: #435a6f 0px 0px 1px 0px;
  padding: 16px 24px 24px;
}
.title {
  font-weight: 500;
  text-align: left;
}
.input {
  padding: 7px 12px;
  width: calc(50vw - 260px);
}
.button {
  background: #5ead80;
  border-radius: 4px;
  color: #fff;
  padding: 8px 16px;
  width: fit-content;
  cursor: pointer;
  font-size: 14px;
  font-weight: 500;
  border-width: 0;
  transition: all 0.2s;
}
.button:hover {
  background: #62b687;
}
.button:active {
  background: forestgreen;
}
.button:focus {
  opacity: 0.8;
}
button:disabled {
  background: #eee;
  pointer-events: none;
}
.button-light {
  border-radius: 4px;
  background: transparent;
  color: #5ead80;
  padding: 8px 16px;
  width: fit-content;
  cursor: pointer;
  font-size: 14px;
  font-weight: 500;
  transition: all 0.2s;
  border: 1px #5ead80 solid;
}
.button-light:hover {
  color: #62b687;
  border-color: #62b687;
}
.button-light:active {
  color: forestgreen;
  border-color: forestgreen;
}
.button-light:focus {
  opacity: 0.8;
}
.row {
  display: flex;
  justify-content: space-between;
}
.item-container {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-start;
}
.item-placeholder {
  height: 408px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 16px;
}
.item {
  background: #ddd;
  box-shadow: #435a6f 0px 0px 1px 0px;
  border-radius: 4px;
  padding: 4px 8px;
  box-sizing: border-box;
  width: 30%;
  margin: 4px 1.6%;
  font-size: 13px;
}
.item-selected {
  background: forestgreen;
  color: #fff !important;
  font-weight: 600;
}
</style>
