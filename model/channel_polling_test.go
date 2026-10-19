package model

import (
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/QuantumNous/new-api/common"
	"github.com/QuantumNous/new-api/constant"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func setMemoryCacheForTest(t *testing.T, enabled bool) {
	t.Helper()
	previousEnabled := common.MemoryCacheEnabled
	channelSyncLock.Lock()
	previousChannels := channelsIDM
	previousGroups := group2model2channels
	previousAdvanced := channel2advancedCustomConfig
	channelSyncLock.Unlock()

	common.MemoryCacheEnabled = enabled
	t.Cleanup(func() {
		common.MemoryCacheEnabled = previousEnabled
		channelSyncLock.Lock()
		channelsIDM = previousChannels
		group2model2channels = previousGroups
		channel2advancedCustomConfig = previousAdvanced
		channelSyncLock.Unlock()
	})
}

func insertPollingChannel(t *testing.T, keys []string) *Channel {
	t.Helper()
	channel := &Channel{
		Name:   "polling-cursor",
		Key:    strings.Join(keys, "\n"),
		Status: common.ChannelStatusEnabled,
		Group:  "default",
		Models: "gpt-test",
		ChannelInfo: ChannelInfo{
			IsMultiKey:           true,
			MultiKeySize:         len(keys),
			MultiKeyMode:         constant.MultiKeyModePolling,
			MultiKeyPollingIndex: 0,
		},
	}
	require.NoError(t, DB.Create(channel).Error)
	require.NoError(t, DB.Create(&Ability{
		Group:     channel.Group,
		Model:     channel.Models,
		ChannelId: channel.Id,
		Enabled:   true,
	}).Error)
	t.Cleanup(func() {
		channelPollingCursors.Delete(channel.Id)
		require.NoError(t, DB.Where("channel_id = ?", channel.Id).Delete(&Ability{}).Error)
		require.NoError(t, DB.Delete(&Channel{}, channel.Id).Error)
	})
	return channel
}

func TestPollingCursorSurvivesInitChannelCache(t *testing.T) {
	setMemoryCacheForTest(t, true)

	twoKey := insertPollingChannel(t, []string{"k0", "k1"})
	InitChannelCache()
	first, err := GetChannelById(twoKey.Id, true)
	require.NoError(t, err)
	key, idx, apiErr := first.GetNextEnabledKey()
	require.Nil(t, apiErr)
	require.Equal(t, "k0", key)
	require.Equal(t, 0, idx)
	InitChannelCache()
	second, err := GetChannelById(twoKey.Id, true)
	require.NoError(t, err)
	key, idx, apiErr = second.GetNextEnabledKey()
	require.Nil(t, apiErr)
	require.Equal(t, "k1", key)
	require.Equal(t, 1, idx)

	var storedTwo Channel
	require.NoError(t, DB.First(&storedTwo, twoKey.Id).Error)
	assert.Equal(t, 0, storedTwo.ChannelInfo.MultiKeyPollingIndex)

	channel := insertPollingChannel(t, []string{"k0", "k1", "k2", "k3"})
	InitChannelCache()

	cached, err := CacheGetChannel(channel.Id)
	require.NoError(t, err)
	key, idx, apiErr = cached.GetNextEnabledKey()
	require.Nil(t, apiErr)
	require.Equal(t, "k0", key)
	require.Equal(t, 0, idx)
	key, idx, apiErr = cached.GetNextEnabledKey()
	require.Nil(t, apiErr)
	require.Equal(t, "k1", key)
	require.Equal(t, 1, idx)

	InitChannelCache()
	cached, err = CacheGetChannel(channel.Id)
	require.NoError(t, err)
	key, idx, apiErr = cached.GetNextEnabledKey()
	require.Nil(t, apiErr)
	require.Equal(t, "k2", key)
	require.Equal(t, 2, idx)

	var stored Channel
	require.NoError(t, DB.First(&stored, channel.Id).Error)
	assert.Equal(t, 0, stored.ChannelInfo.MultiKeyPollingIndex)

	loaded, err := GetChannelById(channel.Id, true)
	require.NoError(t, err)
	key, idx, apiErr = loaded.GetNextEnabledKey()
	require.Nil(t, apiErr)
	require.Equal(t, "k3", key)
	require.Equal(t, 3, idx)

	cached, err = CacheGetChannel(channel.Id)
	require.NoError(t, err)
	key, idx, apiErr = cached.GetNextEnabledKey()
	require.Nil(t, apiErr)
	require.Equal(t, "k0", key)
	require.Equal(t, 0, idx)

	require.NoError(t, DB.First(&stored, channel.Id).Error)
	assert.Equal(t, 0, stored.ChannelInfo.MultiKeyPollingIndex)
}

func TestPollingCursorConcurrentInitChannelCache(t *testing.T) {
	setMemoryCacheForTest(t, true)
	channel := insertPollingChannel(t, []string{"k0", "k1", "k2"})
	InitChannelCache()

	var counts [3]atomic.Int32
	errCh := make(chan error, 32)
	var wg sync.WaitGroup
	for _, n := range []int{4, 4, 4, 4, 4, 4, 3, 3} {
		wg.Go(func() {
			for range n {
				selected, err := CacheGetChannel(channel.Id)
				if err != nil {
					errCh <- err
					return
				}
				_, idx, apiErr := selected.GetNextEnabledKey()
				if apiErr != nil {
					errCh <- apiErr
					return
				}
				if idx < 0 || idx >= len(counts) {
					errCh <- fmt.Errorf("polling index %d out of range", idx)
					return
				}
				counts[idx].Add(1)
			}
		})
	}
	for range 2 {
		wg.Go(func() {
			for range 5 {
				InitChannelCache()
			}
		})
	}
	wg.Wait()
	close(errCh)
	for err := range errCh {
		require.NoError(t, err)
	}
	assert.Equal(t, int32(10), counts[0].Load())
	assert.Equal(t, int32(10), counts[1].Load())
	assert.Equal(t, int32(10), counts[2].Load())
}

func TestPollingDoesNotRestoreDisabledKey(t *testing.T) {
	setMemoryCacheForTest(t, false)
	channel := insertPollingChannel(t, []string{"key-a", "key-b"})

	snapshot, err := GetChannelById(channel.Id, true)
	require.NoError(t, err)

	changed := UpdateChannelStatus(channel.Id, "key-a", common.ChannelStatusAutoDisabled, "provider rejected key")
	require.True(t, changed)

	_, _, apiErr := snapshot.GetNextEnabledKey()
	require.Nil(t, apiErr)

	fresh, err := GetChannelById(channel.Id, true)
	require.NoError(t, err)
	assert.Equal(t, common.ChannelStatusAutoDisabled, fresh.ChannelInfo.MultiKeyStatusList[0])
}
