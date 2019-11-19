// Copyright 2015 The go-ethereum Authors
// This file is part of the go-ethereum library.
//
// The go-ethereum library is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// The go-ethereum library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with the go-ethereum library. If not, see <http://www.gnu.org/licenses/>.

// Package fetcher contains the block announcement based synchronisation.
package fetcher

import (
	"errors"
	"math/rand"
	"time"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/common/prque"
	"github.com/ethereum/go-ethereum/consensus"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/log"
)

const (
	arriveTimeout = 500 * time.Millisecond // Time allowance before an announced block is explicitly requested
	gatherSlack   = 100 * time.Millisecond // Interval used to collate almost-expired announces with fetches
	fetchTimeout  = 5 * time.Second        // Maximum allotted time to return an explicitly requested block
	maxUncleDist  = 7                      // Maximum allowed backward distance from the chain head
	maxQueueDist  = 32                     // Maximum allowed distance from the chain head to queue
	hashLimit     = 256                    // Maximum number of unique blocks a peer may have announced
	blockLimit    = 64                     // Maximum number of unique blocks a peer may have delivered
)

var (
	errTerminated = errors.New("terminated")
)

// blockRetrievalFn is a callback type for retrieving a block from the local chain.
type blockRetrievalFn func(common.Hash) *types.Block

// headerRequesterFn is a callback type for sending a header retrieval request.
type headerRequesterFn func(common.Hash) error

// bodyRequesterFn is a callback type for sending a body retrieval request.
type bodyRequesterFn func([]common.Hash) error

// headerVerifierFn is a callback type to verify a block's header for fast propagation.
type headerVerifierFn func(header *types.Header) error

// blockBroadcasterFn is a callback type for broadcasting a block to connected peers.
type blockBroadcasterFn func(block *types.Block, propagate bool)

// chainHeightFn is a callback type to retrieve the current chain height.
type chainHeightFn func() uint64

// chainInsertFn is a callback type to insert a batch of blocks into the local chain.
type chainInsertFn func(types.Blocks) (int, error)

// peerDropFn is a callback type for dropping a peer detected as malicious.
type peerDropFn func(id string)

// announce is the hash notification of the availability of a new block in the
// network.
type announce struct {
	hash   common.Hash   // Hash of the block being announced
	number uint64        // Number of the block being announced (0 = unknown | old protocol)
	header *types.Header // Header of the block partially reassembled (new protocol)
	time   time.Time     // Timestamp of the announcement

	origin string // Identifier of the peer originating the notification

	fetchHeader headerRequesterFn // Fetcher function to retrieve the header of an announced block
	fetchBodies bodyRequesterFn   // Fetcher function to retrieve the body of an announced block
}

// headerFilterTask represents a batch of headers needing fetcher filtering.
type headerFilterTask struct {
	peer    string          // The source peer of block headers
	headers []*types.Header // Collection of headers to filter
	time    time.Time       // Arrival time of the headers
}

// bodyFilterTask represents a batch of block bodies (transactions and uncles)
// needing fetcher filtering.
type bodyFilterTask struct {
	peer         string                 // The source peer of block bodies
	transactions [][]*types.Transaction // Collection of transactions per block bodies
	uncles       [][]*types.Header      // Collection of uncles per block bodies
	time         time.Time              // Arrival time of the blocks' contents
}

// inject represents a schedules import operation.
type inject struct {
	origin string
	block  *types.Block
}

// Fetcher is responsible for accumulating block announcements from various peers
// and scheduling them for retrieval.
type Fetcher struct {
	// Various event channels
	notify chan *announce
	inject chan *inject

	headerFilter chan chan *headerFilterTask
	bodyFilter   chan chan *bodyFilterTask

	done chan common.Hash
	quit chan struct{}

	// Announce states
	announces  map[string]int              // Per peer announce counts to prevent memory exhaustion
	announced  map[common.Hash][]*announce // Announced blocks, scheduled for fetching
	fetching   map[common.Hash]*announce   // Announced blocks, currently fetching
	fetched    map[common.Hash][]*announce // Blocks with headers fetched, scheduled for body retrieval
	completing map[common.Hash]*announce   // Blocks with headers, currently body-completing

	// Block cache
	queue  *prque.Prque            // Queue containing the import operations (block number sorted)
	queues map[string]int          // Per peer block counts to prevent memory exhaustion
	queued map[common.Hash]*inject // Set of already queued blocks (to dedupe imports)

	// Callbacks
	getBlock       blockRetrievalFn   // Retrieves a block from the local chain
	verifyHeader   headerVerifierFn   // Checks if a block's headers have a valid proof of work
	broadcastBlock blockBroadcasterFn // Broadcasts a block to connected peers
	chainHeight    chainHeightFn      // Retrieves the current chain's height
	insertChain    chainInsertFn      // Injects a batch of blocks into the chain
	dropPeer       peerDropFn         // Drops a peer for misbehaving

	// Testing hooks
	announceChangeHook func(common.Hash, bool) // Method to call upon adding or deleting a hash from the announce list
	queueChangeHook    func(common.Hash, bool) // Method to call upon adding or deleting a block from the import queue
	fetchingHook       func([]common.Hash)     // Method to call upon starting a block (eth/61) or header (eth/62) fetch
	completingHook     func([]common.Hash)     // Method to call upon starting a block body fetch (eth/62)
	importedHook       func(*types.Block)      // Method to call upon successful block import (both eth/61 and eth/62)
}

// New creates a block fetcher to retrieve blocks based on hash announcements.
func New(getBlock blockRetrievalFn, verifyHeader headerVerifierFn, broadcastBlock blockBroadcasterFn, chainHeight chainHeightFn, insertChain chainInsertFn, dropPeer peerDropFn) *Fetcher {
	return &Fetcher{
		notify:         make(chan *announce),
		inject:         make(chan *inject),
		headerFilter:   make(chan chan *headerFilterTask),
		bodyFilter:     make(chan chan *bodyFilterTask),
		done:           make(chan common.Hash),
		quit:           make(chan struct{}),
		announces:      make(map[string]int),
		announced:      make(map[common.Hash][]*announce),
		fetching:       make(map[common.Hash]*announce),
		fetched:        make(map[common.Hash][]*announce),
		completing:     make(map[common.Hash]*announce),
		queue:          prque.New(nil),
		queues:         make(map[string]int),
		queued:         make(map[common.Hash]*inject),
		getBlock:       getBlock,
		verifyHeader:   verifyHeader,
		broadcastBlock: broadcastBlock,
		chainHeight:    chainHeight,
		insertChain:    insertChain,
		dropPeer:       dropPeer,
	}
}

// Start boots up the announcement based synchroniser, accepting and processing
// hash notifications and block fetches until termination requested.
func (f *Fetcher) Start() {
	go f.loop()
}

// Stop terminates the announcement based synchroniser, canceling all pending
// operations.
func (f *Fetcher) Stop() {
	close(f.quit)
}

// Notify announces the fetcher of the potential availability of a new block in
// the network.
// 造了一个 announce 结构，并将其发送给了 Fetcher.notify 这个 channel。
func (f *Fetcher) Notify(peer string, hash common.Hash, number uint64, time time.Time,
	headerFetcher headerRequesterFn, bodyFetcher bodyRequesterFn) error {
	block := &announce{
		hash:   hash,
		number: number,
		time:   time,
		origin: peer,
		// 下载Header
		fetchHeader: headerFetcher,
		// 下载Body
		fetchBodies: bodyFetcher,
	}
	select {
	case f.notify <- block:
		return nil
	case <-f.quit:
		return errTerminated
	}
}

// Enqueue tries to fill gaps the fetcher's future import queue.
func (f *Fetcher) Enqueue(peer string, block *types.Block) error {
	op := &inject{
		origin: peer,
		block:  block,
	}
	select {
	case f.inject <- op:
		return nil
	case <-f.quit:
		return errTerminated
	}
}

// FilterHeaders extracts all the headers that were explicitly requested by the fetcher,
// returning those that should be handled differently.
// 处理下载的header消息
func (f *Fetcher) FilterHeaders(peer string, headers []*types.Header, time time.Time) []*types.Header {
	log.Trace("Filtering headers", "peer", peer, "headers", len(headers))

	// Send the filter channel to the fetcher
	filter := make(chan *headerFilterTask)

	select {
	// 先发一个通信用的 channel 给 headerFilter
	case f.headerFilter <- filter:
	case <-f.quit:
		return nil
	}
	// Request the filtering of the header list
	// 将要过滤的 header 发送给 filter
	select {
	case filter <- &headerFilterTask{peer: peer, headers: headers, time: time}:
	case <-f.quit:
		return nil
	}
	// Retrieve the headers remaining after filtering
	// 再从 filter 中获取过滤结果
	select {
	case task := <-filter:
		return task.headers
	case <-f.quit:
		return nil
	}
}

// FilterBodies extracts all the block bodies that were explicitly requested by
// the fetcher, returning those that should be handled differently.
func (f *Fetcher) FilterBodies(peer string, transactions [][]*types.Transaction, uncles [][]*types.Header, time time.Time) ([][]*types.Transaction, [][]*types.Header) {
	log.Trace("Filtering bodies", "peer", peer, "txs", len(transactions), "uncles", len(uncles))

	// Send the filter channel to the fetcher
	filter := make(chan *bodyFilterTask)

	select {
	case f.bodyFilter <- filter:
	case <-f.quit:
		return nil, nil
	}
	// Request the filtering of the body list
	select {
	case filter <- &bodyFilterTask{peer: peer, transactions: transactions, uncles: uncles, time: time}:
	case <-f.quit:
		return nil, nil
	}
	// Retrieve the bodies remaining after filtering
	select {
	case task := <-filter:
		return task.transactions, task.uncles
	case <-f.quit:
		return nil, nil
	}
}

// Loop is the main fetcher loop, checking and processing various notification
// events.
// Fetcher 主要生命周期
func (f *Fetcher) loop() {
	// Iterate the block fetching until a quit is requested
	fetchTimer := time.NewTimer(0)
	completeTimer := time.NewTimer(0)

	for {
		// Clean up any expired block fetches
		// 去掉超时区块
		for hash, announce := range f.fetching {
			if time.Since(announce.time) > fetchTimeout {
				f.forgetHash(hash)
			}
		}
		// Import any queued blocks that could potentially fit
		height := f.chainHeight()
		for !f.queue.Empty() {
			op := f.queue.PopItem().(*inject)
			hash := op.block.Hash()
			if f.queueChangeHook != nil {
				f.queueChangeHook(hash, false)
			}
			// If too high up the chain or phase, continue later
			// 如果高度大于本地最高高度 + 1，则将信息放回队列并结束入库
			number := op.block.NumberU64()
			if number > height+1 {
				// 优先队列，每次都是弹出高度最小的区块
				// 最小的不符合规则，则不需要继续判断了
				f.queue.Push(op, -int64(number))
				if f.queueChangeHook != nil {
					f.queueChangeHook(hash, true)
				}
				break
			}
			// Otherwise if fresh and still unknown, try and import
			// 确保区块不会太旧并且在本地未存储
			if number+maxUncleDist < height || f.getBlock(hash) != nil {
				f.forgetBlock(hash)
				continue
			}
			// 存入本地数据库
			f.insert(op.origin, op.block)
		}
		// Wait for an outside event to occur
		select {
		case <-f.quit:
			// Fetcher terminating, abort all operations
			return

		case notification := <-f.notify:
			// A block was announced, make sure the peer isn't DOSing us
			propAnnounceInMeter.Mark(1)

			// 判断这个节点已经通知的、但是还未下载成功的哈希的数量。
			count := f.announces[notification.origin] + 1
			// 如果数量太多，就先不去处理最新的通知了。这样即可以避免进程缓存了太多待下载的区块信息而占用过多内存，也可以防止对方节点不断地恶意发送无效区块。
			if count > hashLimit {
				log.Debug("Peer exceeded outstanding announces", "peer", notification.origin, "limit", hashLimit)
				propAnnounceDOSMeter.Mark(1)
				break
			}
			// If we have a valid block number, check that it's potentially useful
			// 确保当前通知的这个区块不会太旧（比本地区块高度小 maxUncleDist）
			// 或太新（比本地区块高度大 maxQueueDist）
			if notification.number > 0 {
				if dist := int64(notification.number) - int64(f.chainHeight()); dist < -maxUncleDist || dist > maxQueueDist {
					log.Debug("Peer discarded announcement", "peer", notification.origin, "number", notification.number, "hash", notification.hash, "distance", dist)
					propAnnounceDropMeter.Mark(1)
					break
				}
			}
			// All is well, schedule the announce if block's not yet downloading
			// 确保当前通知的区块还未开始下载，保证同一个区块没有被重复下载
			if _, ok := f.fetching[notification.hash]; ok {
				break
			}
			if _, ok := f.completing[notification.hash]; ok {
				break
			}
			// 更新节点的待下载量
			f.announces[notification.origin] = count
			f.announced[notification.hash] = append(f.announced[notification.hash], notification)
			if f.announceChangeHook != nil && len(f.announced[notification.hash]) == 1 {
				f.announceChangeHook(notification.hash, true)
			}
			// 最后如果 Fetcher.announced 中只有刚才新加入的这一个区块信息
			if len(f.announced) == 1 {
				f.rescheduleFetch(fetchTimer)
			}

		case op := <-f.inject:
			// A direct block insertion was requested, try and fill any pending gaps
			propBroadcastInMeter.Mark(1)
			f.enqueue(op.origin, op.block)

		case hash := <-f.done:
			// A pending import finished, remove all traces of the notification
			f.forgetHash(hash)
			f.forgetBlock(hash)
		// 	下载中
		case <-fetchTimer.C:
			// At least one block's timer ran out, check for needing retrieval
			request := make(map[string][]common.Hash)
			// 选择要下载的区块，从 announced 转移到 fetching 中，
			// 并将要下载的哈希填充到 request 中
			for hash, announces := range f.announced {
				// 「可以下载的」条件是区块通知的时间已经过去了 arriveTimeout-gatherSlack 这么长时间，并将相关信息转移到 Fetcher.fetching 中
				// 区块的状态从「通知」变成了「下载中」
				if time.Since(announces[0].time) > arriveTimeout-gatherSlack {
					// Pick a random peer to retrieve from, reset all others
					announce := announces[rand.Intn(len(announces))]
					f.forgetHash(hash)

					// If the block still didn't arrive, queue for fetching
					if f.getBlock(hash) == nil {
						request[announce.origin] = append(request[announce.origin], hash)
						f.fetching[hash] = announce
					}
				}
			}
			// Send out all block header requests
			// 发送下载 header 的请求
			for peer, hashes := range request {
				log.Trace("Fetching scheduled headers", "peer", peer, "list", hashes)

				// Create a closure of the fetch and schedule in on a new thread
				fetchHeader, hashes := f.fetching[hashes[0]].fetchHeader, hashes
				go func() {
					if f.fetchingHook != nil {
						f.fetchingHook(hashes)
					}
					for _, hash := range hashes {
						headerFetchMeter.Mark(1)
						fetchHeader(hash) // Suboptimal, but protocol doesn't allow batch header retrievals
					}
				}()
			}
			// Schedule the next fetch if blocks are still pending
			// 重新设置下次的下载发起时间
			f.rescheduleFetch(fetchTimer)
		// 	发起body下载请求代码
		case <-completeTimer.C:
			// At least one header's timer ran out, retrieve everything
			request := make(map[string][]common.Hash)

			// 从 Fetcher.fetched 中选取将要下载 body 的是信息放入 request 中
			for hash, announces := range f.fetched {
				// Pick a random peer to retrieve from, reset all others
				announce := announces[rand.Intn(len(announces))]
				f.forgetHash(hash)

				// If the block still didn't arrive, queue for completion
				if f.getBlock(hash) == nil {
					request[announce.origin] = append(request[announce.origin], hash)
					f.completing[hash] = announce
				}
			}
			// Send out all block body requests
			// 发送请求
			for peer, hashes := range request {
				log.Trace("Fetching scheduled bodies", "peer", peer, "list", hashes)

				// Create a closure of the fetch and schedule in on a new thread
				if f.completingHook != nil {
					f.completingHook(hashes)
				}
				bodyFetchMeter.Mark(int64(len(hashes)))
				go f.completing[hashes[0]].fetchBodies(hashes)
			}
			// Schedule the next fetch if blocks are still pending
			f.rescheduleComplete(completeTimer)

		// 	处理下载的header
		case filter := <-f.headerFilter:

			// Headers arrived from a remote peer. Extract those that were explicitly
			// requested by the fetcher, and return everything else so it's delivered
			// to other parts of the system.
			// 从filter 变量中取得要过滤的信息 task
			var task *headerFilterTask
			select {
			case task = <-filter:
			case <-f.quit:
				return
			}
			headerFilterInMeter.Mark(int64(len(task.headers)))

			// Split the batch of headers into unknown ones (to return to the caller),
			// known incomplete ones (requiring body retrievals) and completed blocks.
			// unknown 代表「未知」，这些区块根本不是 Fetcher 对象发起下载的；
			// incomplete 代表是 Fetcher 发起下载的区块，但这里只有 header 数据，还需少 body 数据；
			// complete 也代表是 Fetcher 发起的区块，并且这个区块已经不缺数据可以直接导入本地数据库了。
			// complete 状态的数据都是空块，因为空区块的 body 为空，不需要下载。
			unknown, incomplete, complete := []*types.Header{}, []*announce{}, []*types.Block{}
			for _, header := range task.headers {
				hash := header.Hash()

				// Filter fetcher-requested headers from other synchronisation algorithms
				// 判断是否是我们正在下载的 header
				if announce := f.fetching[hash]; announce != nil && announce.origin == task.peer && f.fetched[hash] == nil && f.completing[hash] == nil && f.queued[hash] == nil {
					// If the delivered header does not match the promised number, drop the announcer
					if header.Number.Uint64() != announce.number {
						log.Trace("Invalid block number fetched", "peer", announce.origin, "hash", header.Hash(), "announced", announce.number, "provided", header.Number)
						f.dropPeer(announce.origin)
						f.forgetHash(hash)
						continue
					}
					// Only keep if not imported by other means
					// 判断此区块在本地是否已存在
					// 为在 Fetcher 发起下载的过程中，downloader 模块可能已经将其下载完成了
					if f.getBlock(hash) == nil {
						announce.header = header
						announce.time = task.time

						// If the block is empty (header only), short circuit into the final import queue
						// 判断是否是空区块。
						// 对于空区块，直接加入到 Fetcher.completing 中
						if header.TxHash == types.DeriveSha(types.Transactions{}) && header.UncleHash == types.CalcUncleHash([]*types.Header{}) {
							log.Trace("Block empty, skipping body retrieval", "peer", announce.origin, "number", header.Number, "hash", header.Hash())

							block := types.NewBlockWithHeader(header)
							block.ReceivedAt = task.time

							complete = append(complete, block)
							f.completing[hash] = announce
							continue
						}
						// Otherwise add to the list of blocks needing completion
						// 非空区块，保存在 incomplete 中
						incomplete = append(incomplete, announce)
					} else {
						log.Trace("Block already imported, discarding header", "peer", announce.origin, "number", header.Number, "hash", header.Hash())
						// 本地已存在这个区块
						f.forgetHash(hash)
					}
				} else {
					// Fetcher doesn't know about it, add to the return list
					// 不是我们想要下载的区块
					unknown = append(unknown, header)
				}
			}
			headerFilterOutMeter.Mark(int64(len(unknown)))
			select {
			// 将 unknown 通知 Fetcher.FilterHeaders
			case filter <- &headerFilterTask{headers: unknown, time: task.time}:
			case <-f.quit:
				return
			}
			// Schedule the retrieved headers for body completion
			// 将 incomplete 加入到 Fetcher.fetched 中
			for _, announce := range incomplete {
				hash := announce.header.Hash()
				if _, ok := f.completing[hash]; ok {
					continue
				}
				f.fetched[hash] = append(f.fetched[hash], announce)
				if len(f.fetched) == 1 {
					// 重置定时器
					f.rescheduleComplete(completeTimer)
				}
			}
			// Schedule the header-only blocks for import
			// 将 complete 加入等待入库的队列中
			for _, block := range complete {
				if announce := f.completing[block.Hash()]; announce != nil {
					f.enqueue(announce.origin, block)
				}
			}

		case filter := <-f.bodyFilter:
			// Block bodies arrived, extract any explicitly requested blocks, return the rest
			var task *bodyFilterTask
			select {
			case task = <-filter:
			case <-f.quit:
				return
			}
			bodyFilterInMeter.Mark(int64(len(task.transactions)))

			blocks := []*types.Block{}

			// 对于每一组数据，查看其 tx 哈希 和 uncle 哈希是否在 Fetcher.queued 中的 header 中
			for i := 0; i < len(task.transactions) && i < len(task.uncles); i++ {
				// Match up a body to any possible completion request
				matched := false

				for hash, announce := range f.completing {

					if f.queued[hash] == nil {
						// 对于 task 中的每一组数据，计算它们的 transaction 哈希和 uncle 哈希
						txnHash := types.DeriveSha(types.Transactions(task.transactions[i]))
						uncleHash := types.CalcUncleHash(task.uncles[i])

						// 判定是否是我们下载的数据
						if txnHash == announce.header.TxHash && uncleHash == announce.header.UncleHash && announce.origin == task.peer {
							// Mark the body matched, reassemble if still unknown
							// 确定是我们发起的下载数据
							matched = true

							if f.getBlock(hash) == nil {
								block := types.NewBlockWithHeader(announce.header).WithBody(task.transactions[i], task.uncles[i])
								block.ReceivedAt = task.time

								blocks = append(blocks, block)
							} else {
								f.forgetHash(hash)
							}
						}
					}
				}
				if matched {
					task.transactions = append(task.transactions[:i], task.transactions[i+1:]...)
					task.uncles = append(task.uncles[:i], task.uncles[i+1:]...)
					i--
					continue
				}
			}

			bodyFilterOutMeter.Mark(int64(len(task.transactions)))
			select {
			// 留在 task 中的数据返回给 Fetcher.FilterBodies
			case filter <- task:
			case <-f.quit:
				return
			}
			// Schedule the retrieved blocks for ordered import
			// 调用 Fetcher.enqueue 将 blocks 中的所有区块加入到待入库的队列 Fetcher.queue
			for _, block := range blocks {
				if announce := f.completing[block.Hash()]; announce != nil {
					f.enqueue(announce.origin, block)
				}
			}
		}
	}
}

// rescheduleFetch resets the specified fetch timer to the next announce timeout.
// 调整下次fetch的时间
func (f *Fetcher) rescheduleFetch(fetch *time.Timer) {
	// Short circuit if no blocks are announced
	if len(f.announced) == 0 {
		return
	}
	// Otherwise find the earliest expiring announcement
	// 在通知区块中找到距当前时间最近的通知
	earliest := time.Now()
	for _, announces := range f.announced {
		if earliest.After(announces[0].time) {
			earliest = announces[0].time
		}
	}
	fetch.Reset(arriveTimeout - time.Since(earliest))
}

// rescheduleComplete resets the specified completion timer to the next fetch timeout.
func (f *Fetcher) rescheduleComplete(complete *time.Timer) {
	// Short circuit if no headers are fetched
	if len(f.fetched) == 0 {
		return
	}
	// Otherwise find the earliest expiring announcement
	earliest := time.Now()
	for _, announces := range f.fetched {
		if earliest.After(announces[0].time) {
			earliest = announces[0].time
		}
	}
	complete.Reset(gatherSlack - time.Since(earliest))
}

// enqueue schedules a new future import operation, if the block to be imported
// has not yet been seen.
func (f *Fetcher) enqueue(peer string, block *types.Block) {
	hash := block.Hash()

	// Ensure the peer isn't DOSing us
	// 确保不会有太多的来自同一个节点的区块等待入库
	count := f.queues[peer] + 1
	if count > blockLimit {
		log.Debug("Discarded propagated block, exceeded allowance", "peer", peer, "number", block.Number(), "hash", hash, "limit", blockLimit)
		propBroadcastDOSMeter.Mark(1)
		f.forgetHash(hash)
		return
	}

	// Discard any past or too distant blocks
	// 确保这个区块不会太久也不会太新
	if dist := int64(block.NumberU64()) - int64(f.chainHeight()); dist < -maxUncleDist || dist > maxQueueDist {
		log.Debug("Discarded propagated block, too far away", "peer", peer, "number", block.Number(), "hash", hash, "distance", dist)
		propBroadcastDropMeter.Mark(1)
		f.forgetHash(hash)
		return
	}

	// Schedule the block for future importing
	// 最后将区块数据放到了 Fetcher.queue 中。
	// 注意 Fetcher.queue 是一个优先级队列，它的优先级为高度的负值，所以在 Pop 操作中，高度越小的元素优先级越高，越会被首先弹出。
	if _, ok := f.queued[hash]; !ok {
		op := &inject{
			origin: peer,
			block:  block,
		}
		f.queues[peer] = count
		f.queued[hash] = op
		f.queue.Push(op, -int64(block.NumberU64()))
		if f.queueChangeHook != nil {
			f.queueChangeHook(op.block.Hash(), true)
		}
		log.Debug("Queued propagated block", "peer", peer, "number", block.Number(), "hash", hash, "queued", f.queue.Size())
	}
}

// insert spawns a new goroutine to run a block insertion into the chain. If the
// block's number is at the same height as the current import phase, it updates
// the phase states accordingly.
// 插入区块
// 将区块查询到本地之前，广播将要插入的区块
func (f *Fetcher) insert(peer string, block *types.Block) {
	hash := block.Hash()

	// Run the import on a new thread
	log.Debug("Importing propagated block", "peer", peer, "number", block.Number(), "hash", hash)
	go func() {
		defer func() { f.done <- hash }()

		// If the parent's unknown, abort insertion
		parent := f.getBlock(block.ParentHash())
		if parent == nil {
			log.Debug("Unknown parent of propagated block", "peer", peer, "number", block.Number(), "hash", hash, "parent", block.ParentHash())
			return
		}
		// Quickly validate the header and propagate the block if it passes
		switch err := f.verifyHeader(block.Header()); err {
		case nil:
			// All ok, quickly propagate to our peers
			propBroadcastOutTimer.UpdateSince(block.ReceivedAt)
			go f.broadcastBlock(block, true)

		case consensus.ErrFutureBlock:
			// Weird future block, don't fail, but neither propagate

		default:
			// Something went very wrong, drop the peer
			log.Debug("Propagated block verification failed", "peer", peer, "number", block.Number(), "hash", hash, "err", err)
			f.dropPeer(peer)
			return
		}
		// Run the actual import and log any issues
		if _, err := f.insertChain(types.Blocks{block}); err != nil {
			log.Debug("Propagated block import failed", "peer", peer, "number", block.Number(), "hash", hash, "err", err)
			return
		}
		// If import succeeded, broadcast the block
		propAnnounceOutTimer.UpdateSince(block.ReceivedAt)
		go f.broadcastBlock(block, false)

		// Invoke the testing hook if needed
		if f.importedHook != nil {
			f.importedHook(block)
		}
	}()
}

// forgetHash removes all traces of a block announcement from the fetcher's
// internal state.
func (f *Fetcher) forgetHash(hash common.Hash) {
	// Remove all pending announces and decrement DOS counters
	for _, announce := range f.announced[hash] {
		f.announces[announce.origin]--
		if f.announces[announce.origin] <= 0 {
			delete(f.announces, announce.origin)
		}
	}
	delete(f.announced, hash)
	if f.announceChangeHook != nil {
		f.announceChangeHook(hash, false)
	}
	// Remove any pending fetches and decrement the DOS counters
	if announce := f.fetching[hash]; announce != nil {
		f.announces[announce.origin]--
		if f.announces[announce.origin] <= 0 {
			delete(f.announces, announce.origin)
		}
		delete(f.fetching, hash)
	}

	// Remove any pending completion requests and decrement the DOS counters
	for _, announce := range f.fetched[hash] {
		f.announces[announce.origin]--
		if f.announces[announce.origin] <= 0 {
			delete(f.announces, announce.origin)
		}
	}
	delete(f.fetched, hash)

	// Remove any pending completions and decrement the DOS counters
	if announce := f.completing[hash]; announce != nil {
		f.announces[announce.origin]--
		if f.announces[announce.origin] <= 0 {
			delete(f.announces, announce.origin)
		}
		delete(f.completing, hash)
	}
}

// forgetBlock removes all traces of a queued block from the fetcher's internal
// state.
func (f *Fetcher) forgetBlock(hash common.Hash) {
	if insert := f.queued[hash]; insert != nil {
		f.queues[insert.origin]--
		if f.queues[insert.origin] == 0 {
			delete(f.queues, insert.origin)
		}
		delete(f.queued, hash)
	}
}
