# Logic

Note: For openai embeddings we can use dot products because the vectors are already normalized

## Phase 1: Filtering

[] Find what sectors and what profile types the user wants to connect with from context_builder_management
[] Find the profile type to which the user belongs to

[] Filter possible candidates for this user by

- Do they have common sectors selected in context builder?
- Are they complementary? Bidirectional complementary is what we are after. i.e. Do candidate belong to an organisation in which the user has opted in looking_to_connect or does user's organisation matches what the candidate is looking_to_connect
  Startup seeking investors + Investor seeking startups = Strong match (both directions)
  Startup seeking investors + Investor seeking corporates = Weak match (one direction)
  Startup seeking investors + Another startup seeking investors = No match (neither direction)

Conclusion: We will have a list of user_ids that are compatible to our input user_id

## Phase 2: Embeddings

- Given the list from the previous phase check if there are embeddings for them already generated
- Lazy evaluation of profile embeddings and caching them on demand - NEEDS CLARITY
- Calcualte assymetric similarity, i.e. we cant directly use dot product of embeddings so lets jugaad by
  - A_to_B: Dot product of A's intent with B's org, representing what A wants vs B offers
  - B_to_A: Dot procuct of B's intent with A's org, representing what B wants vs A offers
