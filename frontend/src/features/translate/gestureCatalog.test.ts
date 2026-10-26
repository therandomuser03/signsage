import { test } from "node:test";
import assert from "node:assert/strict";
import {
  GESTURE_CATALOG,
  buildGestureGroups,
  filterGestureEntries,
} from "./gestureCatalog";

test("alphabetical groups keep names ordered from A to Z", () => {
  const groups = buildGestureGroups(GESTURE_CATALOG);

  assert.ok(groups.length > 0);
  assert.deepEqual(
    groups.map((group) => group.letter),
    ["F", "H", "M", "N", "P", "S", "T", "W", "Y"],
  );
  assert.deepEqual(groups[0].entries.map((entry) => entry.label), ["Food"]);
  assert.deepEqual(groups[1].entries.map((entry) => entry.label), ["Hello", "Help", "How"]);
});

test("search is case-insensitive and matches key metadata", () => {
  assert.deepEqual(
    filterGestureEntries("yes", GESTURE_CATALOG).map((entry) => entry.label),
    ["Yes"],
  );
  assert.deepEqual(
    filterGestureEntries("point", GESTURE_CATALOG).map((entry) => entry.label),
    ["You", "Me"],
  );
  assert.deepEqual(
    filterGestureEntries("greeting", GESTURE_CATALOG).map((entry) => entry.label),
    ["Hello"],
  );
});

test("empty searches stay empty and the drawer shows no matches", () => {
  const matches = filterGestureEntries("zzzz-no-such-sign", GESTURE_CATALOG);
  assert.deepEqual(matches, []);

  const groups = buildGestureGroups(matches);
  assert.deepEqual(groups, []);
});

test("drawer state distinguishes live support from browse-only entries", () => {
  const live = GESTURE_CATALOG.filter((entry) => entry.status === "live-supported").map((entry) => entry.label);
  const browseOnly = GESTURE_CATALOG.filter((entry) => entry.status === "browse-only").map((entry) => entry.label);

  assert.ok(live.includes("Hello"));
  assert.ok(browseOnly.includes("Food"));
  assert.ok(live.length > 0);
  assert.ok(browseOnly.length > 0);
});
