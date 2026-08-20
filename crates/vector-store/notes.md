## DataProvider

Should *reuse* the same to_internal_id() and to_external_id() as inmem-provider. (is it possible to override traits?). We aspire to have the same Id semantics as inmem provider. VS PrimaryId and whatever diskann overlords decided will be good for internal id - rbitrary id mapping.

## Execution context

DefaultContext should probably be ok? Or we can use empty context like inmem (just reuse).

## SetElement

This one we need to write ourselves. But we can probably just impl `SetElement<&[f32]>` - this will be fine for the foreseeable future.
But the I don't really get how the implemenation here should look like. Like we still need to allocate an internal id, but we just dont want to store the vector in RAM? Do we just cut out the in RAM saving part? IDK

## Guard

Noop will be fine for now since inmem also uses this.

## Search Accessor

Here we can have some fun. Since we always provide a singular starting point (at least for now). We can make everything very simple. We can even override provided methods to not require the HashMap as we will have a singular starting points.

## expand_beam

This is by far the hardest thing to understand.  Filtered ann is out of the question for now.

## PruneAccessor

I don't really know here. This is very tricky to get right. Let's chat about this.

## NeighbourAccessor

Whatever works in inmem should be fine. This is about the adjecency if i'm not mistaken.

## Delete

Soft deletes should be fine ig. If it's deleted from DB then we will not even be able to fetch it back from DB so ig it might be possible to even make it a no-op, but that's a bit dangerous and should be considered later.
Hard deletes should be considered in the future as this might save some memory ig.

## Post Processing

I feel like majority of this can be safely reused?

## Errors

Should behave the same way inmem does.

## Other notes

This should be as minimal as possible. No multi inserts. No crazy optimizations. Database requests can also be optimized later.

## Takeaways

Reuse where possible. First let's create the provider and later worry about not having a handle to a scylla session (do not do it for now).

# After claude implementation

## HybridBackend

Look good, might move mod doc to here. Will likely require db_index or a reader that will allow to query ScyllaDB.

## NoVectors

If we want to reuse InmemProvider, then I guess this is the way too go. Dimension validation is already done in vector-store so not sure if necessary here, will  need to check that (TODO).
Make the explanation as to why it's like this clearer.

## NoDistance

This is all just basically a part of a hack to access the neighbours list kept by the InmemProvider via PruneStrategy (we don't actaully use the InmemPruningStrategy) so we can use it ourselves since it's private. This is very much a hack and something to look at closely (TODO).

## HybridProvider + Inflight

I guess this all makes sense. The worst part is by far the fact that we can't easily access the start_id of the frozen vector (this is very much prone to regressions on future releases). But I guess the start_id is needed for the HybridSearchAccessor to work.

The `load()` function looks OK in general but I am yet to understand why the output looks like it does. Don't fully understand `fill()` and `expand_beam()` which was supposed to be the only hard part.

`adjacency()` really is a crazy hack that does not look nice and is really hard to understand, but I genuinely think it works.

### verify_start_slot

Yet another hack that already described earlier. This shit crazy is all I can say.

## DataProvider impl

It's good, nothing spectacular here.

## SetElement

I actually think it's pretty simple if all used functions work correctly.
Reusing the diskann noop Guard id into our Guard id is pretty smart actually.
But what I don't get is why keep the vector in the guard if everything here looks sync. I think it's because if someone wants to read this vector immedietaly after insert then it might not be possible to fetch from scylla? Something along these lines.

## Delete

Pretty straightforward other than `release()` which requires an additional verification (TODO)

## InflightGuard

Pretty basic guard, it works correctly. Question is if it's necessary.

# Accessors

This is by far the hardest to understand part. I did not go through this fully, but it does kinda make sense.

# Takeaways

There is so much hacky stuff here that I must also check out how difficult it would be to not reuse the Id translation and adjacency or write a totally separate Provider. Question - how bad would it be to just copy the inmem impl? Does it bring any legal consequences?
