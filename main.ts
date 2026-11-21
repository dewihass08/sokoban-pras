// sokoban-solver-01 entry: the level is a hallway with independent rooms, so each room is solved on its own with a
// push-macro A* (box squares + the player's reachable region), pruned by dead squares, frozen 2x2 blocks and a static
// push-distance table, and guided by the minimum-cost assignment of boxes to goals over push distances. Rooms' optima
// add up to the level's, so the answer is push-optimal whenever each room search proves its optimum. Budgets are counts
// of expanded states, never the clock: two runs answer the same bytes.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

type Level = { w: number; h: number; wall: Uint8Array; goals: number[]; boxes: number[]; player: number };

const DX = [0, 0, -1, 1];
const DY = [-1, 1, 0, 0];
const LOWER = ['u', 'd', 'l', 'r'];
const UPPER = ['U', 'D', 'L', 'R'];

/** The most states one room search expands before it gives up (work, not the clock). */
const ROOM_STATES = 500_000;
/** The most states the whole set's searches expand together, and one room's greedy fallback. */
const TOTAL_STATES = 4_000_000;
const FALLBACK_STATES = 200_000;

function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
  return { w, h, wall, goals, boxes: boxes.sort((a, b) => a - b), player };
}

/** The squares from which a box alone can be pushed onto some goal: pulled back from the goals over open floor. */
function liveSquares(level: Level): Uint8Array {
  const { w, wall, goals } = level;
  const live = new Uint8Array(wall.length);
  const queue = goals.slice();
  for (const g of goals) live[g] = 1;
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi];
    for (let k = 0; k < 4; k++) {
      const y = x - (DY[k] * w + DX[k]);
      if (!wall[y] && !wall[y - (DY[k] * w + DX[k])] && !live[y]) {
        live[y] = 1;
        queue.push(y);
      }
    }
  }
  return live;
}

/** For one goal, the pushes a box alone needs from every square to reach it (a pull back from the goal). */
function pushDistancesTo(level: Level, goal: number): Int32Array {
  const { w, wall } = level;
  const dist = new Int32Array(wall.length).fill(-1);
  dist[goal] = 0;
  const queue = [goal];
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi];
    for (let k = 0; k < 4; k++) {
      const d = DY[k] * w + DX[k];
      const y = x - d;
      if (wall[y] || wall[y - d] || dist[y] >= 0) continue;
      dist[y] = dist[x] + 1;
      queue.push(y);
    }
  }
  return dist;
}

/** Whether a box just pushed to `t` sits in a frozen 2x2 block of walls and boxes with a box off its goal. */
function frozenBlock(level: Level, occupied: Uint8Array, goal: Uint8Array, t: number): boolean {
  const { w, wall } = level;
  for (let k = 0; k < 4; k++) {
    const a = k === 0 ? t : k === 1 ? t - 1 : k === 2 ? t - w : t - w - 1;
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (!(wall[a] || occupied[a]) || !(wall[b] || occupied[b]) || !(wall[c] || occupied[c]) || !(wall[d] || occupied[d])) continue;
    if ((occupied[a] && !goal[a]) || (occupied[b] && !goal[b]) || (occupied[c] && !goal[c]) || (occupied[d] && !goal[d])) return true;
  }
  return false;
}

/** The squares the player reaches, marked in `seen` with `stamp`; returns the lowest of them (the region's name). */
function region(level: Level, occupied: Uint8Array, from: number, seen: Int32Array, stamp: number, stack: Int32Array): number {
  const { w, wall } = level;
  let low = from;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n];
    if (p < low) low = p;
    for (let k = 0; k < 4; k++) {
      const q = p + DY[k] * w + DX[k];
      if (wall[q] === 0 && occupied[q] === 0 && seen[q] !== stamp) {
        seen[q] = stamp;
        stack[n++] = q;
      }
    }
  }
  return low;
}

/** The shortest walk from `from` to `to` around walls and boxes, in lower LURD (first direction first on ties). */
function walkTo(level: Level, occupied: Uint8Array, from: number, to: number): string | null {
  const { w, wall } = level;
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how: number[] = new Array(wall.length).fill(0);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi];
    if (p === to) break;
    for (let k = 0; k < 4; k++) {
      const q = p + DY[k] * w + DX[k];
      if (wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = k;
      queue.push(q);
    }
  }
  if (prev[to] === -1) return null;
  let s = '';
  for (let p = to; p !== from; p = prev[p]) s = LOWER[how[p]] + s;
  return s;
}

/** All orders of 0..n-1, for the box-to-goal assignment (n is at most four here). */
function permutations(n: number): number[][] {
  const out: number[][] = [];
  const cur: number[] = [];
  const used: boolean[] = new Array(n).fill(false);
  const walk = (): void => {
    if (cur.length === n) {
      out.push(cur.slice());
      return;
    }
    for (let i = 0; i < n; i++) {
      if (used[i]) continue;
      used[i] = true;
      cur.push(i);
      walk();
      cur.pop();
      used[i] = false;
    }
  };
  walk();
  return out;
}

type Budget = { left: number };

/**
 * The push-macro A* of one (sub)level: a state is the boxes' squares and the player's reachable region, a move is one
 * push, and the estimate is the cheapest assignment of boxes to goals over per-goal push distances. `weight` above 1
 * trades optimality for reaching a solution at all.
 */
function solvePushes(level: Level, cap: number, weight: number, budget: Budget): { pushes: number; moves: string } | null {
  const { w, wall, goals, boxes: startBoxes, player: startPlayer } = level;
  const size = wall.length;
  const n = startBoxes.length;
  if (n === 0) return { pushes: 0, moves: '' };
  if (goals.length !== n) return null;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const dists: Int32Array[] = goals.map((g) => pushDistancesTo(level, g));
  for (const b of startBoxes) {
    let any = false;
    for (const dd of dists) if (dd[b] >= 0) any = true;
    if (!any) return null;
  }
  const perms = permutations(n);
  const hCache = new Map<string, number>();
  const estimate = (bs: number[]): number => {
    const k = bs.join(',');
    const hit = hCache.get(k);
    if (hit !== undefined) return hit;
    let best = Infinity;
    for (const p of perms) {
      let sum = 0;
      for (let i = 0; i < n; i++) {
        const d = dists[p[i]][bs[i]];
        if (d < 0) {
          sum = Infinity;
          break;
        }
        sum += d;
      }
      if (sum < best) best = sum;
    }
    hCache.set(k, best);
    return best;
  };
  const occupied = new Uint8Array(size);
  const seenA = new Int32Array(size);
  const seenB = new Int32Array(size);
  const stack = new Int32Array(size);
  let stampA = 0;
  let stampB = 0;

  const boxesOf: number[][] = [];
  const playerOf: number[] = [];
  const gOf: number[] = [];
  const parentOf: number[] = [];
  const fromOf: number[] = [];
  const dirOf: number[] = [];
  const keys: string[] = [];
  const best = new Map<string, number>();
  const heap: number[][] = [];
  const less = (a: number[], b: number[]): boolean => a[0] < b[0] || (a[0] === b[0] && (a[1] < b[1] || (a[1] === b[1] && a[2] < b[2])));
  const pushEntry = (e: number[]): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i], heap[p])) break;
      const t = heap[i];
      heap[i] = heap[p];
      heap[p] = t;
      i = p;
    }
  };
  const popEntry = (): number[] => {
    const top = heap[0];
    const last = heap.pop() as number[];
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1], heap[c])) c++;
        if (!less(heap[c], heap[i])) break;
        const t = heap[i];
        heap[i] = heap[c];
        heap[c] = t;
        i = c;
      }
    }
    return top;
  };

  for (const b of startBoxes) occupied[b] = 1;
  stampA++;
  const low0 = region(level, occupied, startPlayer, seenA, stampA, stack);
  for (const b of startBoxes) occupied[b] = 0;
  const key0 = low0 + ':' + startBoxes.join(',');
  boxesOf.push(startBoxes);
  playerOf.push(startPlayer);
  gOf.push(0);
  parentOf.push(-1);
  fromOf.push(-1);
  dirOf.push(-1);
  keys.push(key0);
  best.set(key0, 0);
  const h0 = estimate(startBoxes);
  pushEntry([h0 * weight, h0, 0, 0]);
  let seq = 1;
  let expanded = 0;

  let found = -1;
  while (heap.length > 0) {
    const top = popEntry();
    const ni = top[3];
    const g = gOf[ni];
    if (best.get(keys[ni]) !== g) continue;
    const bs = boxesOf[ni];
    let solved = true;
    for (const b of bs) if (!goal[b]) solved = false;
    if (solved) {
      found = ni;
      break;
    }
    if (expanded >= cap || budget.left <= 0) return null;
    expanded++;
    budget.left--;
    for (const b of bs) occupied[b] = 1;
    stampA++;
    const here = stampA;
    region(level, occupied, playerOf[ni], seenA, stampA, stack);
    for (let bi = 0; bi < n; bi++) {
      const b = bs[bi];
      for (let k = 0; k < 4; k++) {
        const d = DY[k] * w + DX[k];
        const t = b + d;
        const s = b - d;
        if (wall[t] || occupied[t] || seenA[s] !== here) continue;
        let anyGoal = false;
        for (const dd of dists) if (dd[t] >= 0) anyGoal = true;
        if (!anyGoal) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        const frozen = frozenBlock(level, occupied, goal, t);
        let nb: number[] | null = null;
        let nrep = -1;
        if (!frozen) {
          nb = bs.slice();
          nb[bi] = t;
          nb.sort((a, b2) => a - b2);
          stampB++;
          nrep = region(level, occupied, b, seenB, stampB, stack);
        }
        occupied[t] = 0;
        occupied[b] = 1;
        if (frozen || nb === null) continue;
        const nkey = nrep + ':' + nb.join(',');
        const seenG = best.get(nkey);
        if (seenG !== undefined && seenG <= g + 1) continue;
        const h = estimate(nb);
        if (!Number.isFinite(h)) continue;
        const idx = boxesOf.length;
        boxesOf.push(nb);
        playerOf.push(b);
        gOf.push(g + 1);
        parentOf.push(ni);
        fromOf.push(b);
        dirOf.push(k);
        keys.push(nkey);
        best.set(nkey, g + 1);
        pushEntry([(g + 1) + weight * h, h, seq++, idx]);
      }
    }
    for (const b of bs) occupied[b] = 0;
  }
  if (found < 0) return null;

  // The push chain, then the moves: walk to each push's square, then push.
  const chain: number[] = [];
  for (let s = found; s > 0; s = parentOf[s]) chain.push(s);
  chain.reverse();
  let moves = '';
  let player = startPlayer;
  let boxes = startBoxes;
  for (const s of chain) {
    for (const b of boxes) occupied[b] = 1;
    const from = fromOf[s];
    const k = dirOf[s];
    const d = DY[k] * w + DX[k];
    const walk = walkTo(level, occupied, player, from - d);
    for (const b of boxes) occupied[b] = 0;
    if (walk === null) return null;
    moves += walk + UPPER[k];
    player = from;
    boxes = boxesOf[s];
  }
  return { pushes: chain.length, moves };
}

/**
 * The level's rooms: the dead squares of the player's row (the hallway: a box there reaches no goal) cut out of the
 * floor, and each part left that holds a box a room. Null when the level is not a set of rooms.
 */
function roomsOf(level: Level): { goals: number[]; boxes: number[] }[] | null {
  const { w, wall, goals, boxes, player } = level;
  const live = liveSquares(level);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) if (!wall[i] && !live[i]) cut[i] = 1;
  for (const b of boxes) if (cut[b]) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms: { squares: number[]; goals: number[]; boxes: number[] }[] = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (let k = 0; k < 4; k++) {
        const q = squares[i] + DY[k] * w + DX[k];
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ squares, goals: [], boxes: [] });
  }
  for (const b of boxes) rooms[part[b]].boxes.push(b);
  for (const g of goals) if (part[g] >= 0) rooms[part[g]].goals.push(g);
  if (goals.some((g) => part[g] < 0)) return null;
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

/** The player's square after `moves` (case aside) from `from`. */
function endOf(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves.toLowerCase()) p += ch === 'u' ? -w : ch === 'd' ? w : ch === 'l' ? -1 : 1;
  return p;
}

/** Replays moves on the level: whether every step is legal, the pushes made and whether every box ends on a goal. */
function replay(level: Level, moves: string): { ok: boolean; pushes: number; done: boolean } {
  const { w, wall } = level;
  const boxes = new Set(level.boxes);
  const goal = new Set(level.goals);
  let player = level.player;
  let pushes = 0;
  for (const raw of moves) {
    const ch = raw.toLowerCase();
    const k = ch === 'u' ? 0 : ch === 'd' ? 1 : ch === 'l' ? 2 : ch === 'r' ? 3 : -1;
    if (k < 0) return { ok: false, pushes, done: false };
    const d = DY[k] * w + DX[k];
    const q = player + d;
    if (wall[q]) return { ok: false, pushes, done: false };
    if (boxes.has(q)) {
      const r = q + d;
      if (wall[r] || boxes.has(r)) return { ok: false, pushes, done: false };
      boxes.delete(q);
      boxes.add(r);
      pushes++;
    }
    player = q;
  }
  let done = true;
  for (const b of boxes) if (!goal.has(b)) done = false;
  return { ok: true, pushes, done };
}

function solve(text: string): string {
  const level = parseLevel(text);
  const budget: Budget = { left: TOTAL_STATES };
  const rooms = roomsOf(level);
  if (rooms !== null) {
    let player = level.player;
    let moves = '';
    let ok = true;
    const done: boolean[] = new Array(rooms.length).fill(false);
    for (let ri = 0; ri < rooms.length && ok; ri++) {
      const room = rooms[ri];
      const wall = level.wall.slice();
      for (let oi = 0; oi < rooms.length; oi++) {
        if (oi === ri) continue;
        const other = rooms[oi];
        const blocks = done[oi] ? other.goals : other.boxes;
        for (const b of blocks) wall[b] = 1;
      }
      const sub: Level = {
        w: level.w,
        h: level.h,
        wall,
        goals: room.goals.slice().sort((a, b) => a - b),
        boxes: room.boxes.slice().sort((a, b) => a - b),
        player,
      };
      let solved = solvePushes(sub, ROOM_STATES, 1, budget);
      if (solved === null) solved = solvePushes(sub, FALLBACK_STATES, 3, budget);
      if (solved === null) {
        ok = false;
        break;
      }
      moves += solved.moves;
      player = endOf(level.w, player, solved.moves);
      done[ri] = true;
    }
    if (ok) {
      const check = replay(level, moves);
      if (check.ok && check.done) return moves;
    }
  }
  const whole = solvePushes(level, FALLBACK_STATES, 1, budget) ?? solvePushes(level, FALLBACK_STATES, 3, budget);
  if (whole !== null) {
    const check = replay(level, whole.moves);
    if (check.ok && check.done) return whole.moves;
  }
  return '';
}

for (const id of readdirSync(I).sort()) {
  const moves = solve(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
