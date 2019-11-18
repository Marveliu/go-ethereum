// Copyright 2017 The go-ethereum Authors
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

package downloader

import (
	"fmt"
	"hash"
	"sync"
	"time"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/core/rawdb"
	"github.com/ethereum/go-ethereum/core/state"
	"github.com/ethereum/go-ethereum/ethdb"
	"github.com/ethereum/go-ethereum/log"
	"github.com/ethereum/go-ethereum/trie"
	"golang.org/x/crypto/sha3"
)

// stateReq represents a batch of state fetch requests grouped together into
// a single data retrieval network packet.
type stateReq struct {
	items    []common.Hash              // Hashes of the state items to download
	tasks    map[common.Hash]*stateTask // Download tasks to track previous attempts
	timeout  time.Duration              // Maximum round trip time for this to complete
	timer    *time.Timer                // Timer to fire when the RTT timeout expires
	peer     *peerConnection            // Peer that we're requesting from
	response [][]byte                   // Response data of the peer (nil for timeouts)
	dropped  bool                       // Flag whether the peer dropped off early
}

// timedOut returns if this request timed out.
func (req *stateReq) timedOut() bool {
	return req.response == nil
}

// stateSyncStats is a collection of progress stats to report during a state trie
// sync to RPC requests as well as to display in user logs.
type stateSyncStats struct {
	processed  uint64 // Number of state entries processed
	duplicate  uint64 // Number of state entries downloaded twice
	unexpected uint64 // Number of non-requested state entries received
	pending    uint64 // Number of still pending state entries
}

// syncState starts downloading state with the given root hash.
// 给定根Hash进行同步
func (d *Downloader) syncState(root common.Hash) *stateSync {
	// Create the state sync
	s := newStateSync(d, root)
	select {
	// 新的对象发送给 Downloader.stateSyncStart。最后返回这个新的对象。
	// stateFetcher
	case d.stateSyncStart <- s:
	case <-d.quitCh:
		s.err = errCancelStateFetch
		close(s.done)
	}
	return s
}

// stateFetcher manages the active state sync and accepts requests
// on its behalf.
func (d *Downloader) stateFetcher() {
	for {
		select {
		case s := <-d.stateSyncStart:
			// 进行同步
			for next := s; next != nil; {
				next = d.runStateSync(next)
			}
		case <-d.stateCh:
			// Ignore state responses while no sync is running.
		case <-d.quitCh:
			return
		}
	}
}

// runStateSync runs a state synchronisation until
// it completes or another root
// hash is requested to be switched over to.
// 执行状态同步
func (d *Downloader) runStateSync(s *stateSync) *stateSync {

	var (
		// 正在处理中
		active = make(map[string]*stateReq) // Currently in-flight requests
		// 已经完成的 不管成功或者失败
		finished []*stateReq // Completed or failed requests
		// 超时
		timeout = make(chan *stateReq) // Timed out active requests
	)

	defer func() {
		// Cancel active request timers on exit. Also set peers to idle so they're
		// available for the next sync.
		for _, req := range active {
			req.timer.Stop()
			req.peer.SetNodeDataIdle(len(req.items))
		}
	}()

	// 其实代码这么写是为了在同步某个 state 数据的同时，能够及时发现新的 state 同步请求并进行处理，而处理方式就是停掉之前的同步进程
	// Run the state sync.
	go s.run()
	// 方法退出之前，停止同步的动作
	defer s.Cancel()

	// Listen for peer departure events to cancel assigned tasks
	peerDrop := make(chan *peerConnection, 1024)
	peerSub := s.d.peers.SubscribePeerDrops(peerDrop)
	defer peerSub.Unsubscribe()

	for {
		// Enable sending of the first buffered element if there is one.
		var (
			deliverReq   *stateReq
			deliverReqCh chan *stateReq
		)
		if len(finished) > 0 {
			deliverReq = finished[0]
			deliverReqCh = s.deliver
		}

		select {

		// The stateSync lifecycle:
		// 同一时间只有一个区块的 state 数据被同步；如果要同步新的 state，需要先停掉旧的同步过程。
		// 此处收到新的stateSync 直接返回，执行 defer s.Cancel()
		case next := <-d.stateSyncStart:
			return next

		// 	区块下载完成
		case <-s.done:
			return nil

		// Send the next finished request to the current sync:
		// finished 集合中的第一个已完成的请求就会发送给 stateSync.deliver
		case deliverReqCh <- deliverReq:
			// Shift out the first request, but also set the emptied slot to nil for GC
			copy(finished, finished[1:])
			finished[len(finished)-1] = nil
			finished = finished[:len(finished)-1]

		// Handle incoming state packs:
		// 功接收到数据
		case pack := <-d.stateCh:

			// Discard any data not requested (or previously timed out)
			// 代码首先判断是否在 active 中。如果在则将返回的数据放到 req.response 中，
			// 并将 req 写入 finished 中，然后从 active中删除。
			req := active[pack.PeerId()]
			if req == nil {
				log.Debug("Unrequested node data", "peer", pack.PeerId(), "len", pack.Items())
				continue
			}
			// Finalize the request and queue up for processing
			req.timer.Stop()
			req.response = pack.(*statePack).states

			finished = append(finished, req)
			delete(active, pack.PeerId())

		// Handle dropped peer connections:
		// 处理节点断开连接的情况
		case p := <-peerDrop:
			// Skip if no request is currently pending
			req := active[p.id]
			if req == nil {
				continue
			}
			// Finalize the request and queue up for processing
			req.timer.Stop()
			req.dropped = true

			finished = append(finished, req)
			delete(active, p.id)

		// Handle timed-out requests:
		case req := <-timeout:
			// If the peer is already requesting something else, ignore the stale timeout.
			// This can happen when the timeout and the delivery happens simultaneously,
			// causing both pathways to trigger.
			// 如果超时的请求在 active 中，则将其从 active 中删除，并将其加入到 finished 中（完成但失败了）。
			if active[req.peer.id] != req {
				continue
			}
			// Move the timed out data back into the download queue
			finished = append(finished, req)
			delete(active, req.peer.id)

		// Track outgoing state requests:
		// 上代表 stateSync 对象发起了一次请求
		case req := <-d.trackStateReq:
			// If an active request already exists for this peer, we have a problem. In
			// theory the trie node schedule must never assign two requests to the same
			// peer. In practice however, a peer might receive a request, disconnect and
			// immediately reconnect before the previous times out. In this case the first
			// request is never honored, alas we must not silently overwrite it, as that
			// causes valid requests to go missing and sync to get stuck.
			// 首先通过 active 判断这个节点是否有正在处理的请求，如果是则将旧的请求中断并加入finished中（完成但失败了）。
			// 然后为新的 req 设置一个 timer 后，将 req 加入到 active 中。
			if old := active[req.peer.id]; old != nil {
				log.Warn("Busy peer assigned new state fetch", "peer", old.peer.id)

				// Make sure the previous one doesn't get siletly lost
				old.timer.Stop()
				old.dropped = true

				finished = append(finished, old)
			}
			// Start a timer to notify the sync loop if the peer stalled.
			req.timer = time.AfterFunc(req.timeout, func() {
				select {
				case timeout <- req:
				case <-s.done:
					// Prevent leaking of timer goroutines in the unlikely case where a
					// timer is fired just before exiting runStateSync.
				}
			})
			active[req.peer.id] = req
		}
	}
}

// stateSync schedules requests for downloading a particular state trie defined
// by a given state root.
// 因为所谓的 state 数据，实际上就是一棵 trie 树，而同步过的过程，就是在同步这棵树上的每一个节点。
// 在最开始的时候，downloader 只有 trie 树的根哈希（Root），使用这个哈希同步到的就是树的根节点。
// 那么接下来需要同步的节点的哈希，就需要 trie.Sync 帮忙从这个根结点解析出来（trie 的每个节点会记录子节点的哈希）
// 然后 downloader 才可以使用这些哈希去同步相应的节点。依此不断进行，直到最底层的叶子节点同完成，整个 trie 树就同步完成了。
type stateSync struct {
	d *Downloader // Downloader instance to access and manage current peerset

	sched  *trie.Sync                 // State trie sync scheduler defining the tasks
	keccak hash.Hash                  // Keccak256 hasher to verify deliveries with
	tasks  map[common.Hash]*stateTask // Set of tasks currently queued for retrieval

	numUncommitted   int
	bytesUncommitted int

	deliver    chan *stateReq // Delivery channel multiplexing peer responses
	cancel     chan struct{}  // Channel to signal a termination request
	cancelOnce sync.Once      // Ensures cancel only ever gets called once
	done       chan struct{}  // Channel to signal termination completion
	err        error          // Any error hit during sync (set before completion)
}

// stateTask represents a single trie node download task, containing a set of
// peers already attempted retrieval from to detect stalled syncs and abort.
type stateTask struct {
	attempts map[string]struct{}
}

// newStateSync creates a new state trie download scheduler. This method does not
// yet start the sync. The user needs to call run to initiate.
func newStateSync(d *Downloader, root common.Hash) *stateSync {
	return &stateSync{
		d:       d,
		sched:   state.NewStateSync(root, d.stateDB, d.stateBloom),
		keccak:  sha3.NewLegacyKeccak256(),
		tasks:   make(map[common.Hash]*stateTask),
		deliver: make(chan *stateReq),
		cancel:  make(chan struct{}),
		done:    make(chan struct{}),
	}
}

// run starts the task assignment and response processing loop, blocking until
// it finishes, and finally notifying any goroutines waiting for the loop to
// finish.
func (s *stateSync) run() {
	s.err = s.loop()
	close(s.done)
}

// Wait blocks until the sync is done or canceled.
func (s *stateSync) Wait() error {
	<-s.done
	return s.err
}

// Cancel cancels the sync and waits until it has shut down.
func (s *stateSync) Cancel() error {
	s.cancelOnce.Do(func() { close(s.cancel) })
	return s.Wait()
}

// loop is the main event loop of a state trie sync. It it responsible for the
// assignment of new tasks to peers (including sending it to them) as well as
// for the processing of inbound data. Note, that the loop does not directly
// receive data from peers, rather those are buffered up in the downloader and
// pushed here async. The reason is to decouple processing from data receipt
// and timeouts.
// 同步过程
func (s *stateSync) loop() (err error) {
	// Listen for new peer events to assign tasks to them
	newPeer := make(chan *peerConnection, 1024)
	peerSub := s.d.peers.SubscribeNewPeers(newPeer)
	defer peerSub.Unsubscribe()
	defer func() {
		cerr := s.commit(true)
		if err == nil {
			err = cerr
		}
	}()

	// Keep assigning new tasks until the sync completes or aborts
	for s.sched.Pending() > 0 {
		if err = s.commit(false); err != nil {
			return err
		}
		s.assignTasks()
		// Tasks assigned, wait for something to happen
		select {
		case <-newPeer:
			// New peer arrived, try to assign it download tasks

		case <-s.cancel:
			return errCancelStateFetch

		case <-s.d.cancelCh:
			return errCanceled

		case req := <-s.deliver:
			// 收到了数据
			// Response, disconnect or timeout triggered, drop the peer if stalling
			log.Trace("Received node data response", "peer", req.peer.id, "count", len(req.response), "dropped", req.dropped, "timeout", !req.dropped && req.timedOut())
			// 规则校验
			if len(req.items) <= 2 && !req.dropped && req.timedOut() {
				// 2 items are the minimum requested, if even that times out, we've no use of
				// this peer at the moment.
				log.Warn("Stalling state sync, dropping peer", "peer", req.peer.id)
				if s.d.dropPeer == nil {
					// The dropPeer method is nil when `--copydb` is used for a local copy.
					// Timeouts can occur if e.g. compaction hits at the wrong time, and can be ignored
					req.peer.log.Warn("Downloader wants to drop peer, but peerdrop-function is not set", "peer", req.peer.id)
				} else {
					s.d.dropPeer(req.peer.id)

					// If this peer was the master peer, abort sync immediately
					s.d.cancelLock.RLock()
					master := req.peer.id == s.d.cancelPeer
					s.d.cancelLock.RUnlock()

					if master {
						s.d.cancel()
						return errTimeout
					}
				}
			}
			// Process all the received blobs and check for stale delivery
			delivered, err := s.process(req)
			if err != nil {
				log.Warn("Node data write error", "err", err)
				return err
			}
			req.peer.SetNodeDataIdle(delivered)
		}
	}
	return nil
}

func (s *stateSync) commit(force bool) error {
	if !force && s.bytesUncommitted < ethdb.IdealBatchSize {
		return nil
	}
	start := time.Now()
	b := s.d.stateDB.NewBatch()
	if err := s.sched.Commit(b); err != nil {
		return err
	}
	if err := b.Write(); err != nil {
		return fmt.Errorf("DB write error: %v", err)
	}
	s.updateStats(s.numUncommitted, 0, 0, time.Since(start))
	s.numUncommitted = 0
	s.bytesUncommitted = 0
	return nil
}

// assignTasks attempts to assign new tasks to all idle peers, either from the
// batch currently being retried, or fetching new data from the trie sync itself.
// 发起同步请求
func (s *stateSync) assignTasks() {
	// Iterate over all idle peers and try to assign them state fetches
	peers, _ := s.d.peers.NodeDataIdlePeers()
	for _, p := range peers {
		// Assign a batch of fetches proportional to the estimated latency/bandwidth
		cap := p.NodeDataCapacity(s.d.requestRTT())
		req := &stateReq{peer: p, timeout: s.d.requestTTL()}
		s.fillTasks(cap, req)

		// If the peer was assigned tasks to fetch, send the network request
		if len(req.items) > 0 {
			req.peer.log.Trace("Requesting new batch of data", "type", "state", "count", len(req.items))
			select {
			case s.d.trackStateReq <- req:
				req.peer.FetchNodeData(req.items)
			case <-s.cancel:
			case <-s.d.cancelCh:
			}
		}
	}
}

// fillTasks fills the given request object with a maximum of n state download
// tasks to send to the remote peer.
func (s *stateSync) fillTasks(n int, req *stateReq) {
	// Refill available tasks from the scheduler.
	if len(s.tasks) < n {
		new := s.sched.Missing(n - len(s.tasks))
		for _, hash := range new {
			s.tasks[hash] = &stateTask{make(map[string]struct{})}
		}
	}
	// Find tasks that haven't been tried with the request's peer.
	req.items = make([]common.Hash, 0, n)
	req.tasks = make(map[common.Hash]*stateTask, n)
	for hash, t := range s.tasks {
		// Stop when we've gathered enough requests
		if len(req.items) == n {
			break
		}
		// Skip any requests we've already tried from this peer
		if _, ok := t.attempts[req.peer.id]; ok {
			continue
		}
		// Assign the request to this peer
		t.attempts[req.peer.id] = struct{}{}
		req.items = append(req.items, hash)
		req.tasks[hash] = t
		delete(s.tasks, hash)
	}
}

// process iterates over a batch of delivered state data, injecting each item
// into a running state sync, re-queuing any items that were requested but not
// delivered. Returns whether the peer actually managed to deliver anything of
// value, and any error that occurred.
//
func (s *stateSync) process(req *stateReq) (int, error) {
	// Collect processing stats and update progress if valid data was received
	duplicate, unexpected, successful := 0, 0, 0

	defer func(start time.Time) {
		if duplicate > 0 || unexpected > 0 {
			s.updateStats(0, duplicate, unexpected, time.Since(start))
		}
	}(time.Now())

	// Iterate over all the delivered data and inject one-by-one into the trie
	// 将数据注入到trie中
	for _, blob := range req.response {
		_, hash, err := s.processNodeData(blob)
		switch err {
		case nil:
			s.numUncommitted++
			s.bytesUncommitted += len(blob)
			successful++
		case trie.ErrNotRequested:
			unexpected++
		case trie.ErrAlreadyProcessed:
			duplicate++
		default:
			return successful, fmt.Errorf("invalid state node %s: %v", hash.TerminalString(), err)
		}
		delete(req.tasks, hash)
	}
	// Put unfulfilled tasks back into the retry queue
	npeers := s.d.peers.Len()
	for hash, task := range req.tasks {
		// If the node did deliver something, missing items may be due to a protocol
		// limit or a previous timeout + delayed delivery. Both cases should permit
		// the node to retry the missing items (to avoid single-peer stalls).
		if len(req.response) > 0 || req.timedOut() {
			delete(task.attempts, req.peer.id)
		}
		// If we've requested the node too many times already, it may be a malicious
		// sync where nobody has the right data. Abort.
		if len(task.attempts) >= npeers {
			return successful, fmt.Errorf("state node %s failed with all peers (%d tries, %d peers)", hash.TerminalString(), len(task.attempts), npeers)
		}
		// Missing item, place into the retry queue.
		s.tasks[hash] = task
	}
	return successful, nil
}

// processNodeData tries to inject a trie node data blob delivered from a remote
// peer into the state trie, returning whether anything useful was written or any
// error occurred.
func (s *stateSync) processNodeData(blob []byte) (bool, common.Hash, error) {
	res := trie.SyncResult{Data: blob}
	s.keccak.Reset()
	s.keccak.Write(blob)
	s.keccak.Sum(res.Hash[:0])
	committed, _, err := s.sched.Process([]trie.SyncResult{res})
	return committed, res.Hash, err
}

// updateStats bumps the various state sync progress counters and displays a log
// message for the user to see.
func (s *stateSync) updateStats(written, duplicate, unexpected int, duration time.Duration) {
	s.d.syncStatsLock.Lock()
	defer s.d.syncStatsLock.Unlock()

	s.d.syncStatsState.pending = uint64(s.sched.Pending())
	s.d.syncStatsState.processed += uint64(written)
	s.d.syncStatsState.duplicate += uint64(duplicate)
	s.d.syncStatsState.unexpected += uint64(unexpected)

	if written > 0 || duplicate > 0 || unexpected > 0 {
		log.Info("Imported new state entries", "count", written, "elapsed", common.PrettyDuration(duration), "processed", s.d.syncStatsState.processed, "pending", s.d.syncStatsState.pending, "retry", len(s.tasks), "duplicate", s.d.syncStatsState.duplicate, "unexpected", s.d.syncStatsState.unexpected)
	}
	if written > 0 {
		rawdb.WriteFastTrieProgress(s.d.stateDB, s.d.syncStatsState.processed)
	}
}
