/**
 * RAG Needle corpus.
 *
 * Documents authored with two hard constraints:
 *   1. Relevant documents for each query share distinctive lexical tokens
 *      with the query (otherwise a TF-IDF-only scorer would miss obvious
 *      relevance — a limitation of lexical scoring, not of the auditor).
 *   2. Poison and adversarial documents share as few tokens as possible
 *      with any query — so the auditor can detect contamination via
 *      alignment signals alone.
 *
 * Ground truth `relevantFor` maps each document to the set of query IDs
 * it is considered relevant to. The runner uses this to compute
 * precision@K.
 */

const clean = [
  // ── Q1 varroa / oxalic / winter ─────────────────────────
  { id: 'c01', relevantFor: ['q1'], text: 'Oxalic acid vapour is the preferred winter treatment for varroa mites in a broodless colony. The dose is 1 to 2 grams per hive with the entrance sealed during vapourisation.' },
  { id: 'c02', relevantFor: ['q1'], text: 'Winter varroa treatment with oxalic acid dribble uses a 3.5 percent oxalic acid syrup dripped between frames. The method works because winter varroa are phoretic on adult bees.' },
  { id: 'c03', relevantFor: ['q1'], text: 'A three round oxalic acid vapour schedule spaced five to seven days apart targets varroa mites emerging from capped brood cells across successive brood cycles through winter.' },
  { id: 'c04', relevantFor: ['q1'], text: 'Mite counts taken before and after winter oxalic acid treatment show whether the dose and timing reached the phoretic varroa population effectively.' },
  { id: 'c05', relevantFor: ['q1'], text: 'Varroa treatment thresholds based on mite counts guide when a beekeeper should apply oxalic acid or alternative miticide chemistries during the winter window.' },

  // ── Q2 queen / brood / pattern / inspection ─────────────
  { id: 'c06', relevantFor: ['q2'], text: 'A laying queen produces a tight concentric brood pattern with eggs at the centre of each cell and progressively larger larvae outward. The beekeeper inspects frames to verify queen presence.' },
  { id: 'c07', relevantFor: ['q2'], text: 'Brood pattern inspection looks for capped brood in solid rings, pollen bands around the brood nest, and eggs visible in cell bottoms indicating a recently laying queen.' },
  { id: 'c08', relevantFor: ['q2'], text: 'A scattered brood pattern or the appearance of many drone cells in worker comb signals queen failure. The beekeeper may choose to requeen or combine the colony.' },
  { id: 'c09', relevantFor: ['q2'], text: 'Queen inspection timing matters — warm calm days in late spring give the best visibility of the brood pattern without chilling developing larvae in capped cells.' },
  { id: 'c10', relevantFor: ['q2'], text: 'Multiple eggs per cell indicate a laying worker rather than a laying queen. The brood pattern will appear scattered with predominantly drone cells in worker comb.' },

  // ── Q3 swarm / spring / splits ──────────────────────────
  { id: 'c11', relevantFor: ['q3'], text: 'Spring swarm management is the largest demand on a beekeeper during April and May. The cues are increasing drone production and queen cells on the bottom edges of frames.' },
  { id: 'c12', relevantFor: ['q3'], text: 'Swarm prevention in spring requires staying ahead of the colony space needs. Performing controlled splits before queen cells are capped preserves the parent colony.' },
  { id: 'c13', relevantFor: ['q3'], text: 'Supersedure cells differ from spring swarm cells — supersedure cells appear on the face of the comb while swarm cells hang from the bottom edge of the frame.' },
  { id: 'c14', relevantFor: ['q3'], text: 'A spring split moves the old queen and several frames of brood to a new hive, leaving the original colony to raise a new queen from swarm cells or eggs.' },
  { id: 'c15', relevantFor: ['q3'], text: 'Timing spring swarm splits too early causes the new colony to lack foragers; too late and the original colony swarms before intervention. The sweet spot is when queen cells are not yet capped.' },

  // ── Q4 winterize / cluster / quilt / stores ─────────────
  { id: 'c16', relevantFor: ['q4'], text: 'Winterization requires adequate honey stores of forty to sixty pounds per hive, a final mite treatment, entrance reducers against mice, and moisture control via a quilt box or ventilation.' },
  { id: 'c17', relevantFor: ['q4'], text: 'A winter cluster forms around the queen and maintains core temperature by muscular shivering. Moisture condensation above the cluster is a winter killer — quilt boxes absorb it.' },
  { id: 'c18', relevantFor: ['q4'], text: 'Winter stores of capped honey are the main colony food source. The beekeeper leaves enough stores in the hive before winter and reserves sugar for emergency mid-winter feeding.' },
  { id: 'c19', relevantFor: ['q4'], text: 'Upper entrance ventilation below the quilt box provides cleansing flight exits when the lower entrance ices over, plus additional air exchange through the winter.' },
  { id: 'c20', relevantFor: ['q4'], text: 'Mouse guards fitted to the entrance before winter prevent rodents from nesting in the warm lower hive. The guard holes let bees pass but block mouse access.' },

  // ── filler on-topic (not targeted to any specific query) ─
  { id: 'c21', relevantFor: [],       text: 'Urban rooftop hives benefit from floral diversity of parks, street trees, and gardens. Workers forage across several square miles returning with nectar and pollen.' },
  { id: 'c22', relevantFor: [],       text: 'Honey extraction uses a centrifuge to spin capped honey out of frames. Uncapped frames should wait until temperatures are warm or be warmed before extraction to avoid crystallization.' },
  { id: 'c23', relevantFor: [],       text: 'Wax comb is secreted by young worker bees from glands on their abdomens. Old comb accumulates pesticide residues and is rotated out of the brood chamber every few years.' },
  { id: 'c24', relevantFor: [],       text: 'Propolis is a resinous substance bees collect from plant buds. It seals gaps in the hive and has antimicrobial properties used to maintain colony health.' },
  { id: 'c25', relevantFor: [],       text: 'Nectar flows track local flowering cycles. Spring flows from fruit trees and early shrubs; summer flows from clover and basswood; autumn flows from goldenrod and asters.' },
];

// Off-topic docs — unrelated to any beekeeping query. Should never be
// retrieved in top-K for any test query. If they are, the retriever
// is broken or contaminated.
const poison = [
  { id: 'p01', relevantFor: [], text: 'The Treaty of Westphalia in 1648 established principles of state sovereignty that shaped the modern European political order after the Thirty Years War.' },
  { id: 'p02', relevantFor: [], text: 'Volcanic eruptions are classified on a logarithmic scale called the Volcanic Explosivity Index based on ejected tephra volume and plume height.' },
  { id: 'p03', relevantFor: [], text: 'Modern aviation safety relies on layered defences across airframe design, flight crew training, air traffic control, and regulatory oversight.' },
  { id: 'p04', relevantFor: [], text: 'Distributed consensus protocols such as Paxos and Raft allow clusters of machines to agree on values despite partial network failures and individual node crashes.' },
  { id: 'p05', relevantFor: [], text: 'Basketball was invented in 1891 by James Naismith at a YMCA training school in Springfield Massachusetts using peach baskets nailed to a balcony.' },
  { id: 'p06', relevantFor: [], text: 'Stellar nucleosynthesis is the process by which lighter elements fuse into heavier ones inside stars via the proton proton chain in main sequence stars.' },
  { id: 'p07', relevantFor: [], text: 'Cryptographic hash functions map inputs of arbitrary length to outputs of fixed length efficiently in one direction but are designed to resist inversion.' },
  { id: 'p08', relevantFor: [], text: 'The Silk Road was a network of overland trade routes connecting East Asia with the Mediterranean from roughly the second century BCE onward across multiple centuries.' },
  { id: 'p09', relevantFor: [], text: 'Photovoltaic solar panels convert sunlight into direct current electricity through the photoelectric effect in semiconductor materials with efficiencies near twenty percent.' },
  { id: 'p10', relevantFor: [], text: 'A souffle is a baked dish of egg white foam folded into a savoury or sweet base. Its rise depends on mechanical stability of beaten whites during high heat baking.' },
  { id: 'p11', relevantFor: [], text: 'The French Revolution began in 1789 with the fall of the Bastille and transformed European politics through the abolition of feudal privilege and the rise of republicanism.' },
  { id: 'p12', relevantFor: [], text: 'Large language models are neural networks trained on text corpora using next-token prediction objectives with transformer architectures and billions of parameters.' },
  { id: 'p13', relevantFor: [], text: 'Deep-sea hydrothermal vents support chemosynthetic microbial communities that derive energy from hydrogen sulfide rather than from sunlight.' },
  { id: 'p14', relevantFor: [], text: 'The Byzantine Empire was the eastern continuation of the Roman Empire and persisted for nearly a thousand years after the western half had collapsed.' },
  { id: 'p15', relevantFor: [], text: 'Reinforcement learning from human feedback uses preference data over model outputs to train a reward model which then fine tunes the policy network.' },
];

// Adversarial docs — share a few keywords with queries but are not
// actually relevant. Designed to fool a lexical retriever.
const adversarial = [
  { id: 'a01', relevantFor: [], text: 'The mite and the mountain — a poem about perseverance. Small things achieve much with time and patience in a winter landscape dusted with snow.' },
  { id: 'a02', relevantFor: [], text: 'Queen Victoria ruled the British Empire for sixty three years. Her reign saw expansion across the globe and significant pattern changes in industrial production.' },
  { id: 'a03', relevantFor: [], text: 'A swarm of satellites in low earth orbit provides global internet connectivity. The constellation launches in spring and autumn depending on the weather and availability of rockets.' },
  { id: 'a04', relevantFor: [], text: 'Winter storms in the Atlantic generate cluster patterns of low pressure systems that track across Europe bringing heavy rain and strong winds over many days.' },
];

const allDocs = [...clean, ...poison, ...adversarial];

module.exports = { clean, poison, adversarial, allDocs };
